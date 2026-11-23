import torch
import torch.nn as nn

from basic_gpt.basic_gpt import apply_rope, Block, MLP, ModelConfig
from transformers import PreTrainedModel


# Inherit from HF config base class, which requires defaults and kwargs plumbing
class ModelConfigSwa(ModelConfig):
    model_type = "swa_gpt"
    
    def __init__(
        self,
        swa_window_size: int,
        swa_period: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert swa_window_size >= 1 and swa_period >= 1
        self.swa_window_size = swa_window_size
        self.swa_period = swa_period


class SlidingWindowAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_heads: int, swa_window_size: int):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.swa_window_size = swa_window_size

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
        q = q.reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2) # (B, H, S, head_dim)
        k = k.reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2) # (B, H, S, head_dim)
        v = v.reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2) # (B, H, S, head_dim)

        # RoPE
        q = apply_rope(q)
        k = apply_rope(k)

        # SWA is implemented via attn mask
        # This seems bad since it is still a full matmul, but a real FA tiled kernel will skip the masked work
        attn = q @ k.transpose(-1, -2) / (self.head_dim ** 0.5) # (B, H, S, S)
        mask_upper = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1) # (S, S)
        mask_lower = torch.tril(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=-self.swa_window_size) # (S, S)
        mask = torch.logical_or(mask_upper, mask_lower)
        attn = torch.masked_fill(attn, mask, float("-inf")) # mask (S, S) broadcast -> (1, 1, S, S) -> (B, H, S, S)
        attn = torch.softmax(attn, dim=-1)

        # Output
        out = attn @ v # (B, H, S, D)
        out = out.transpose(1, 2).reshape(B, S, self.head_dim * self.n_heads) # (B, S, D)
        out = self.out(out) # (B, S, input_dim)
        return out


class SwaBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        attn_hidden_dim: int,
        mlp_hidden_dim: int,
        n_heads: int,
        swa_window_size: int
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(input_dim)
        self.attn = SlidingWindowAttention(input_dim, attn_hidden_dim, n_heads, swa_window_size)
        self.norm2 = nn.RMSNorm(input_dim)
        self.mlp = MLP(input_dim, mlp_hidden_dim)

    # x is (B, S, input_dim)
    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# Config supports ratio between SWA and full self attn
class SwaGPT(PreTrainedModel):
    config_class = ModelConfigSwa
    _tied_weights_keys = {
        "lm_head.weight": "embedding.weight",
    }

    def __init__(
        self,
        config: ModelConfigSwa
    ):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.input_dim)
        # nn.Embedding init uses std 1, while nn.Linear init uses std based on Kaiming uniform
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.05)

        # Full self attn every swa_period blocks
        blocks = []
        for idx in range(config.n_blocks):
            if (idx+1) % config.swa_period != 0:
                blocks.append(
                    SwaBlock(
                        config.input_dim,
                        config.attn_hidden_dim,
                        config.mlp_hidden_dim,
                        config.n_heads,
                        config.swa_window_size,
                    )
                )
            else:
                blocks.append(
                    Block(
                        config.input_dim,
                        config.attn_hidden_dim,
                        config.mlp_hidden_dim,
                        config.n_heads,
                    )
                )
        self.blocks = nn.ModuleList(blocks)

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
