import torch
import torch.nn as nn
from transformers import PreTrainedModel
from basic_gpt.basic_gpt import MLP, ModelConfig, MultiHeadAttention


# Inherit from HF config base class, which requires defaults and kwargs plumbing
class ModelConfigHc(ModelConfig):
    model_type = "hc_gpt"
    
    def __init__(
        self,
        n_streams: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_streams = n_streams


# Pre routing: before the attn/mlp, we mix the N residual streams into a single stream input. Thus, we need to learn read strengths
# Post routing: after the attn/mlp, we distribute the single stream output across the N residual streams. Thus, we need to learn write strengths
# Res routing: controls how much of each old stream contributes to each next stream
# H = α * tanh(projection(X)) + b, so we need α, projection, and b
class HyperConnection(nn.Module):
    def __init__(
        self,
        branch: nn.Module,
        dim: int,
        n_streams: int,
        layer_index: int,
        init_alpha: float = 0.01,
    ):
        super().__init__()

        self.branch = branch
        self.branch_norm = nn.RMSNorm(dim)
        self.routing_norm = nn.RMSNorm(dim)

        # Generates one read weight per stream, collapses N streams into 1
        self.pre_proj = nn.Linear(dim, 1, bias=False)
        # Generates one write weight per stream, expands 1 stream into N
        self.post_proj = nn.Linear(dim, 1, bias=False)
        # Generates an N x N matrix allowing pairwise flow from any input stream to any output stream
        self.res_proj = nn.Linear(dim, n_streams, bias=False)

        # Initialize HC as an ordinary residual connections (zeros)
        nn.init.zeros_(self.pre_proj.weight)
        nn.init.zeros_(self.post_proj.weight)
        nn.init.zeros_(self.res_proj.weight)

        # Initialize pre_bias with a simple one-hot round-robin pattern
        pre_bias = torch.zeros(n_streams)
        pre_bias[layer_index % n_streams] = 1.0
        self.pre_bias = nn.Parameter(pre_bias)
        self.post_bias = nn.Parameter(torch.ones(n_streams))
        self.res_bias = nn.Parameter(torch.eye(n_streams))

        self.pre_alpha = nn.Parameter(torch.tensor(init_alpha))
        self.post_alpha = nn.Parameter(torch.tensor(init_alpha))
        self.res_alpha = nn.Parameter(torch.tensor(init_alpha))

    # x is (B, S, n_streams, D)
    def forward(self, x: torch.Tensor):
        routed = self.routing_norm(x)
        
        pre = self.pre_alpha * torch.tanh(      # (B, S, n_streams)
            self.pre_proj(routed).squeeze(-1)
        ) + self.pre_bias
        post = self.post_alpha * torch.tanh(    # (B, S, n_streams)
            self.post_proj(routed).squeeze(-1)
        ) + self.post_bias
        res = self.res_alpha * torch.tanh(      # (B, S, n_streams, n_streams)
            self.res_proj(routed)
        ).transpose(-2, -1) + self.res_bias     

        combined = (pre.unsqueeze(-2) @ x).squeeze(-2)  # (B, S, D)
        combined = self.branch_norm(combined)

        # Branch forward
        out = self.branch(combined) # (B, S, D)

        # Mix streams
        mixed = res @ x # (B, S, n_streams, D)
        distributed = post.unsqueeze(-1) * out.unsqueeze(-2) # (B, S, n_streams, D)

        return mixed + distributed


class HcBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        attn_hidden_dim: int,
        mlp_hidden_dim: int,
        n_heads: int,
        n_streams: int,
        block_index: int,
    ):
        super().__init__()
        self.hc_attention = HyperConnection(
            branch=MultiHeadAttention(input_dim, attn_hidden_dim, n_heads),
            dim=input_dim,
            n_streams=n_streams,
            layer_index=2*block_index,
        )
        self.hc_mlp = HyperConnection(
            branch=MLP(input_dim, mlp_hidden_dim),
            dim=input_dim,
            n_streams=n_streams,
            layer_index=2*block_index+1,
        )

    # x is (B, S, n_streams, input_dim)
    def forward(self, x: torch.Tensor):
        x = self.hc_attention(x)
        x = self.hc_mlp(x)
        return x


# Inherit from HF model class
class HcGPT(PreTrainedModel):
    config_class = ModelConfigHc
    _tied_weights_keys = {
        "lm_head.weight": "embedding.weight",
    }
    
    def __init__(
        self,
        config: ModelConfigHc
    ):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.input_dim)
        # nn.Embedding init uses std 1, while nn.Linear init uses std based on Kaiming uniform
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.05)
        self.blocks = nn.ModuleList([
            HcBlock(
                config.input_dim,
                config.attn_hidden_dim,
                config.mlp_hidden_dim,
                config.n_heads,
                config.n_streams,
                block_index
            ) for block_index in range(config.n_blocks)
        ])
        self.norm = nn.RMSNorm(config.input_dim)
        self.lm_head = nn.Linear(config.input_dim, config.vocab_size, bias=False)
        # Tie weights for less params
        self.lm_head.weight = self.embedding.weight
        self.n_streams = config.n_streams

    # x is (B, S)
    def forward(self, input_ids: torch.Tensor):
        x = self.embedding(input_ids) # (B, S, input_dim)
        
        # Initial expansion via replicating
        x = x.unsqueeze(2).expand(-1, -1, self.n_streams, -1) # (B, S, n_streams, input_dim)
        for block in self.blocks:
            x = block(x)
        
        # Final compression via summing
        x = x.sum(dim=2) # (B, S, input_dim)
        x = self.norm(x)
        logits = self.lm_head(x) # (B, S, V)
        return logits
