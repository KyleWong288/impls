import torch
import torch.nn as nn
from transformers import PreTrainedModel
from basic_gpt.basic_gpt import MLP, ModelConfig, MultiHeadAttention

# Takes raw logits of (B, S, n_streams, n_streams), exponentiates, and alternating 2D normalization
def sinkhorn(logits: torch.Tensor, n_iters):
    matrix = logits.exp()
    for _ in range(n_iters):
        matrix = matrix / matrix.sum(dim=-1, keepdim=True)
        matrix = matrix / matrix.sum(dim=-2, keepdim=True)
    return matrix


# Inherit from HF config base class, which requires defaults and kwargs plumbing
class ModelConfigmHc(ModelConfig):
    model_type = "mhc_gpt"
    
    def __init__(
        self,
        n_streams: int = 4,
        n_sinkhorn_iters: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_streams = n_streams
        self.n_sinkhorn_iters = n_sinkhorn_iters


# Pre routing: H = sigmoid(α * projection(X) + b)
# Post routing: H = 2 * sigmoid(α * projection(X) + b)
# Res routing: H = sinkhorn(α * projection(X) + b)
class ManifoldConstrainedHyperConnection(nn.Module):
    def __init__(
        self,
        branch: nn.Module,
        dim: int,
        n_streams: int,
        n_sinkhorn_iters: int,
        init_alpha: float = 0.01,
        projection_init_std: float = 0.02,
    ):
        super().__init__()

        self.n_streams = n_streams
        self.routing_dim = n_streams * dim
        self.n_sinkhorn_iters = n_sinkhorn_iters

        self.branch = branch
        self.branch_norm = nn.RMSNorm(dim)
        self.routing_norm = nn.RMSNorm(self.routing_dim)

        # Generates one read weight per stream, collapses N streams into 1
        self.pre_proj = nn.Linear(self.routing_dim, n_streams, bias=False)  # (n_streams * dim, n_streams)
        # Generates one write weight per stream, expands 1 stream into N
        self.post_proj = nn.Linear(self.routing_dim, n_streams, bias=False) # (n_streams * dim, n_streams)
        # Generates an N x N matrix allowing pairwise flow from any input stream to any output stream
        self.res_proj = nn.Linear(self.routing_dim, n_streams * n_streams, bias=False) # (n_streams * dim, n_streams * n_streams)

        # Small dynamic routing proposals, gated by alphas initialized to 0.01
        nn.init.normal_(self.pre_proj.weight, mean=0.0, std=projection_init_std)
        nn.init.normal_(self.post_proj.weight, mean=0.0, std=projection_init_std)
        nn.init.normal_(self.res_proj.weight, mean=0.0, std=projection_init_std)

        # These are logits consumed by sigmoid/Sinkhorn, not final routing weights
        self.pre_bias = nn.Parameter(torch.zeros(n_streams))            # (n_streams,)
        self.post_bias = nn.Parameter(torch.zeros(n_streams))           # (n_streams,)
        self.res_bias = nn.Parameter(torch.zeros(n_streams, n_streams)) # (n_streams, n_streams)

        self.pre_alpha = nn.Parameter(torch.tensor(init_alpha))
        self.post_alpha = nn.Parameter(torch.tensor(init_alpha))
        self.res_alpha = nn.Parameter(torch.tensor(init_alpha))

    # x is (B, S, n_streams, D)
    def forward(self, x: torch.Tensor):
        routed = x.flatten(start_dim=-2) # (B, S, n_streams * D)
        routed = self.routing_norm(routed)

        pre = torch.sigmoid(        # (B, S, n_streams)
            self.pre_alpha * self.pre_proj(routed) + self.pre_bias
        )
        post = 2 * torch.sigmoid(   # (B, S, n_streams)
            self.post_alpha * self.post_proj(routed) + self.post_bias
        )
        res = self.res_alpha * self.res_proj(routed)              # (B, S, n_streams * n_streams)
        res = res.unflatten(-1, (self.n_streams, self.n_streams)) # (B, S, n_streams, n_streams)
        res = res + self.res_bias
        res = sinkhorn(res, self.n_sinkhorn_iters)

        combined = (pre.unsqueeze(-2) @ x).squeeze(-2)  # (B, S, D)
        combined = self.branch_norm(combined)

        # Branch forward
        out = self.branch(combined) # (B, S, D)

        # Mix streams
        mixed = res @ x # (B, S, n_streams, D)
        distributed = post.unsqueeze(-1) * out.unsqueeze(-2) # (B, S, n_streams, D)

        return mixed + distributed


class mHcBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        attn_hidden_dim: int,
        mlp_hidden_dim: int,
        n_heads: int,
        n_streams: int,
        n_sinkhorn_iters: int,
    ):
        super().__init__()
        self.hc_attention = ManifoldConstrainedHyperConnection(
            branch=MultiHeadAttention(input_dim, attn_hidden_dim, n_heads),
            dim=input_dim,
            n_streams=n_streams,
            n_sinkhorn_iters=n_sinkhorn_iters,
        )
        self.hc_mlp = ManifoldConstrainedHyperConnection(
            branch=MLP(input_dim, mlp_hidden_dim),
            dim=input_dim,
            n_streams=n_streams,
            n_sinkhorn_iters=n_sinkhorn_iters,
        )

    # x is (B, S, n_streams, input_dim)
    def forward(self, x: torch.Tensor):
        x = self.hc_attention(x)
        x = self.hc_mlp(x)
        return x


# Inherit from HF model class
class mHcGPT(PreTrainedModel):
    config_class = ModelConfigmHc
    _tied_weights_keys = {
        "lm_head.weight": "embedding.weight",
    }
    
    def __init__(
        self,
        config: ModelConfigmHc
    ):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.input_dim)
        # nn.Embedding init uses std 1, while nn.Linear init uses std based on Kaiming uniform
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.05)
        self.blocks = nn.ModuleList([
            mHcBlock(
                config.input_dim,
                config.attn_hidden_dim,
                config.mlp_hidden_dim,
                config.n_heads,
                config.n_streams,
                config.n_sinkhorn_iters,
            ) for _ in range(config.n_blocks)
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
