import torch
import torch.nn as nn
from transformers import PreTrainedModel
from basic_gpt.basic_gpt import MLP, MultiHeadAttention



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

        # H = α * projection(X) + b, so we need α, projection, and b
        self.pre_proj = nn.Linear(dim, 1, bias=False)
        self.post_proj = nn.Linear(dim, 1, bias=False)
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
    pass
