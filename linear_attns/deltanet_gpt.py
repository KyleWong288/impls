import torch
import torch.nn as nn
import torch.nn.functional as F

from basic_gpt.basic_gpt import MLP, ModelConfig
from transformers import PreTrainedModel


# Inherit from HF config base class, which requires defaults and kwargs plumbing
class ModelConfigDN(ModelConfig):
    model_type = "dn_gpt"
    
    def __init__(
        self,
        sc_kernel_size: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sc_kernel_size = sc_kernel_size


class DeltaNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_heads: int, sc_kernel_size: int):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.queries = nn.Linear(input_dim, hidden_dim, bias=False)
        self.keys = nn.Linear(input_dim, hidden_dim, bias=False)
        self.values = nn.Linear(input_dim, hidden_dim, bias=False)
        self.out_norm = nn.RMSNorm(self.head_dim)
        self.out = nn.Linear(hidden_dim, input_dim, bias=False)
        # Write strength is per-head
        self.betas = nn.Linear(input_dim, n_heads, bias=False)
        # Short conv parameters
        self.q_conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=sc_kernel_size,
            groups=hidden_dim,
            bias=False,    
        )
        self.k_conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=sc_kernel_size,
            groups=hidden_dim,
            bias=False,    
        )
        self.v_conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=sc_kernel_size,
            groups=hidden_dim,
            bias=False,    
        )
        self.sc_kernel_size = sc_kernel_size


    # x is (B, S, input_dim)
    def forward(self, x: torch.Tensor):
        B, S = x.shape[0], x.shape[1]
        q = self.queries(x) # (B, S, D)
        k = self.keys(x)    # (B, S, D)
        v = self.values(x)  # (B, S, D)

        # Short conv on time dimension
        padding = self.sc_kernel_size - 1
        q = self.q_conv(F.pad(q.transpose(-1, -2), (padding, 0))).transpose(-1, -2) # (B, S, D)
        k = self.k_conv(F.pad(k.transpose(-1, -2), (padding, 0))).transpose(-1, -2) # (B, S, D)
        v = self.v_conv(F.pad(v.transpose(-1, -2), (padding, 0))).transpose(-1, -2) # (B, S, D)

        # Parallelize across heads
        q = q.reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2) # (B, n_heads, S, head_dim)
        k = k.reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2) # (B, n_heads, S, head_dim)
        v = v.reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2) # (B, n_heads, S, head_dim)

        # DeltaNet doesn't need positive stochastic attn weights
        q = F.normalize(F.silu(q), dim=-1)
        k = F.normalize(F.silu(k), dim=-1)
        v = F.silu(v)

        # Compute write strength
        beta = torch.sigmoid(self.betas(x)) # (B, S, n_heads)
        beta = beta.transpose(1, 2).unsqueeze(-1) # (B, n_heads, S, 1)

        # lazily sequentially compute states due to recurrence relation
        outputs = []
        state = q.new_zeros(B, self.n_heads, self.head_dim, self.head_dim) # (B, n_heads, head_dim, head_dim)
        for t in range(S):
            q_t = q[:, :, t:t+1, :] # (B, n_heads, 1, head_dim)
            k_t = k[:, :, t:t+1, :] # (B, n_heads, 1, head_dim)
            v_t = v[:, :, t:t+1, :] # (B, n_heads, 1, head_dim)
            beta_t = beta[:, :, t:t+1, :] # (B, n_heads, 1, 1)

            predicted_v = k_t @ state # (B, n_heads, 1, head_dim)
            delta = v_t - predicted_v # (B, n_heads, 1, head_dim)
            state = state + beta_t * (k_t.transpose(-1, -2) @ delta) # (B, n_heads, head_dim, head_dim)
            outputs.append(q_t @ state) # (B, n_heads, 1, head_dim)

        attn = torch.cat(outputs, dim=2) # (B, n_heads, S, head_dim)
        attn = self.out_norm(attn)
        attn = attn.transpose(1, 2).reshape(B, S, self.n_heads * self.head_dim) # (B, S, D)
        out = self.out(attn) # (B, S, input_dim)
        return out


class DeltaNetBlock(nn.Module):
    def __init__(self, input_dim: int, attn_hidden_dim: int, mlp_hidden_dim: int, n_heads: int, sc_kernel_size: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(input_dim)
        self.attn = DeltaNet(input_dim, attn_hidden_dim, n_heads, sc_kernel_size)
        self.norm2 = nn.RMSNorm(input_dim)
        self.mlp = MLP(input_dim, mlp_hidden_dim)

    # x is (B, S, input_dim)
    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# DeltaNet from Yang 2024, except naive sequential training
class DeltaNetGPT(PreTrainedModel):
    config_class = ModelConfigDN
    _tied_weights_keys = {
        "lm_head.weight": "embedding.weight",
    }

    def __init__(
        self,
        config: ModelConfigDN
    ):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.input_dim)
        # nn.Embedding init uses std 1, while nn.Linear init uses std based on Kaiming uniform
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.05)
        self.blocks = nn.ModuleList([
            DeltaNetBlock(
                config.input_dim,
                config.attn_hidden_dim,
                config.mlp_hidden_dim,
                config.n_heads,
                config.sc_kernel_size,
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
