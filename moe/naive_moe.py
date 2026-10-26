import torch
import torch.nn as nn

from basic_gpt.basic_gpt import MLP, ModelConfig, MultiHeadAttention
from transformers import PreTrainedModel


class ModelConfigMoE(ModelConfig):
    model_type = "naive_moe_gpt"
    
    def __init__(
        self,
        num_experts: int = 8,
        top_k: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.top_k = top_k


class NaiveMoE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.router = nn.Linear(input_dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [MLP(input_dim, hidden_dim) for _ in range(num_experts)]
        ) 
        self.top_k = top_k
        self.num_experts = num_experts

    # x is (B, S, input_dim)
    def forward(self, x: torch.Tensor):
        batch_size, seq_len, input_dim = x.shape[0], x.shape[1], x.shape[2]
        total_tokens = batch_size * seq_len

        # Compute router probs
        router_scores = self.router(x) # (B, S, num_experts)
        router_probs = router_scores.softmax(dim=-1)

        # Compute selected experts and renormalize
        weights, indices = torch.topk(router_probs, self.top_k, dim=-1) # (B, S, top_k)
        weights = weights / weights.sum(dim=-1, keepdim=True) # (B, S, k)

        # Flatten
        weights = weights.reshape(total_tokens, self.top_k) # (B*S, k)
        indices = indices.reshape(total_tokens, self.top_k) # (B*S, k)
        x_flat = x.reshape(total_tokens, input_dim)         # (B*S, input_dim)
        out_flat = torch.zeros(total_tokens, input_dim, device=x.device, dtype=x.dtype)

        for expert_id, expert in enumerate(self.experts):
            mask = indices == expert_id # (B*S, k)
            # Gives the row id (which token) of each True, and col id (which choice) of each True
            token_ids, choice_ids = mask.nonzero(as_tuple=True) # (N,) where N is the # of True cells

            # Extract the assigned tokens, forward pass, and accumulate
            assigned_tokens = x_flat[token_ids]  # (N, input_dim)
            expert_out = expert(assigned_tokens) # (N, input_dim)
            weighted_out = expert_out * weights[token_ids, choice_ids, None] # (N, input_dim) * (N, 1)

            # index_add() avoids the row copying that advanced indexing with += would create
            out_flat = out_flat.index_add(0, token_ids, weighted_out)
        
        return out_flat.reshape(batch_size, seq_len, input_dim)


class Block(nn.Module):

    def __init__(
        self,
        input_dim: int,
        attn_hidden_dim: int,
        mlp_hidden_dim: int,
        n_heads: int,
        num_experts: int,
        top_k: int
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(input_dim)
        self.attn = MultiHeadAttention(input_dim, attn_hidden_dim, n_heads)
        self.norm2 = nn.LayerNorm(input_dim)
        self.moe = NaiveMoE(input_dim, mlp_hidden_dim, num_experts, top_k)

    # x is (B, S, input_dim)
    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.moe(self.norm2(x))
        return x


# Inherit from HF model class
class NaiveMoEGPT(PreTrainedModel):
    config_class = ModelConfigMoE
    _tied_weights_keys = {
        "lm_head.weight": "embedding.weight",
    }

    def __init__(
        self,
        config: ModelConfigMoE
    ):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.input_dim)
        # nn.Embedding init uses std 1, while nn.Linear init uses std based on Kaiming uniform
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.05)
        self.blocks = nn.ModuleList([
            Block(
                config.input_dim,
                config.attn_hidden_dim,
                config.mlp_hidden_dim,
                config.n_heads,
                config.num_experts,
                config.top_k
            ) for _ in range(config.n_blocks)
        ])
        self.norm = nn.LayerNorm(config.input_dim)
        self.lm_head = nn.Linear(config.input_dim, config.vocab_size, bias=False)
        # Tie weights for less params
        self.lm_head.weight = self.embedding.weight

    # x is (B, S)
    def forward(self, input_ids: torch.Tensor):
        x = self.embedding(input_ids) # (B, S, input_dim)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x) # (B, S, V)
        return logits
