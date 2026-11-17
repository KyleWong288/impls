import torch
import torch.nn as nn

from basic_gpt.basic_gpt import MLP, ModelConfig
from transformers import PreTrainedModel


# Inherit from HF config base class, which requires defaults and kwargs plumbing
class ModelConfigLinearAttn(ModelConfig):
    model_type = "linear_attn_gpt"
    
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)


class LinearAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_heads: int):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.queries = nn.Linear(input_dim, hidden_dim)
        self.keys = nn.Linear(input_dim, hidden_dim)
        self.values = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, input_dim)

    # x is (B, S, input_dim)
    def forward(self, x: torch.Tensor):
        B, S = x.shape[0], x.shape[1]
        q = self.queries(x) # (B, S, D)
        k = self.keys(x)    # (B, S, D)
        v = self.values(x)  # (B, S, D)

        # Parallelize across heads
        q = q.reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2) # (B, n_heads, S, head_dim)
        k = k.reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2) # (B, n_heads, S, head_dim)
        v = v.reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2) # (B, n_heads, S, head_dim)

        # regular attn positive scores: uses softmax exponentiation
        # linear attn positive scores: uses ELU(x) + 1 for this effect instead
        q = torch.nn.functional.elu(q) + 1
        k = torch.nn.functional.elu(k) + 1
        
        # kv states are prefix sum over previous time steps
        kv_outers = k.unsqueeze(-1) * v.unsqueeze(-2) # (B, n_heads, S, head_dim, head_dim)
        kv_states = kv_outers.cumsum(dim=2)           # (B, n_heads, S, head_dim, head_dim)

        # regular attn normalize: softmax divides by the sum of exponentiated q @ k scores
        # linear attn normalize: divides by q dot cumulative k for this effect instead
        k_states = k.cumsum(dim=2)                         # (B, n_heads, S, head_dim)
        qk_sums = (q * k_states).sum(dim=-1, keepdim=True) # (B, n_heads, S, 1)

        attn = (q.unsqueeze(-2) @ kv_states).squeeze(-2) # (B, n_heads, S, head_dim)
        attn = attn / qk_sums.clamp_min(1e-6)            # (B, n_heads, S, head_dim)

        attn = attn.transpose(1, 2).reshape(B, S, self.n_heads * self.head_dim) # (B, S, D)
        out = self.out(attn) # (B, S, input_dim)
        return out


class LinearAttentionBlock(nn.Module):
    def __init__(self, input_dim: int, attn_hidden_dim: int, mlp_hidden_dim: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(input_dim)
        self.attn = LinearAttention(input_dim, attn_hidden_dim, n_heads)
        self.norm2 = nn.RMSNorm(input_dim)
        self.mlp = MLP(input_dim, mlp_hidden_dim)

    # x is (B, S, input_dim)
    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# Linear attention from Katharopoulos et al., 2020 -> (QK)V becomes Q(KV) and avoid softmax
class LinearAttentionGPT(PreTrainedModel):
    config_class = ModelConfigLinearAttn
    _tied_weights_keys = {
        "lm_head.weight": "embedding.weight",
    }

    def __init__(
        self,
        config: ModelConfigLinearAttn
    ):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.input_dim)
        # nn.Embedding init uses std 1, while nn.Linear init uses std based on Kaiming uniform
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.05)
        self.blocks = nn.ModuleList([
            LinearAttentionBlock(
                config.input_dim,
                config.attn_hidden_dim,
                config.mlp_hidden_dim,
                config.n_heads
            ) for _ in range(config.n_blocks)
        ])
        self.norm = nn.RMSNorm(config.input_dim)
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
