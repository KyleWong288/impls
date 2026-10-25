import torch
import torch.nn as nn
from transformers import PreTrainedConfig, PreTrainedModel


# Inherit from HF config base class, which requires defaults and kwargs plumbing
class ModelConfig(PreTrainedConfig):
    model_type = "basic_gpt"

    def __init__(
        self,
        vocab_size: int = 50258,
        input_dim: int = 128,
        attn_hidden_dim: int = 128,
        mlp_hidden_dim: int = 512,
        n_heads: int = 4,
        n_blocks: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.input_dim = input_dim
        self.attn_hidden_dim = attn_hidden_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.n_heads = n_heads
        self.n_blocks = n_blocks


# Expects BHSD format, where D is head_dim
def apply_rope(x: torch.Tensor, rotary_base: float = 10_000):
    assert x.ndim == 4, "apply_rope expects BHSD format"
    seq_len, head_dim = x.shape[-2], x.shape[-1]
    assert head_dim % 2 == 0, "head_dim must be divisible by 2"
    
    positions = torch.arange(seq_len, device=x.device, dtype=torch.float32) # (S,)
    frequencies = 1 / ( # (head_dim // 2,)
        rotary_base ** (torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim)
    )
    angles = torch.outer(positions, frequencies) # (S, head_dim // 2)
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    even = x[:, :, :, ::2] # (B, H, S, head_dim // 2)
    odd = x[:, :, :, 1::2] # (B, H, S, head_dim // 2)
    rotated_even = cos * even - sin * odd
    rotated_odd = sin * even + cos * odd

    out = torch.stack([rotated_even, rotated_odd], dim=-1).flatten(start_dim=-2) # (B, H, S, head_dim // 2, 2) -> (B, H, S, head_dim)
    return out


class MultiHeadAttention(nn.Module):
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
        q = q.reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2) # (B, H, S, head_dim)
        k = k.reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2) # (B, H, S, head_dim)
        v = v.reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2) # (B, H, S, head_dim)

        # RoPE
        q = apply_rope(q)
        k = apply_rope(k)

        # Attention
        attn = q @ k.transpose(-1, -2) / (self.head_dim ** 0.5) # (B, H, S, S)
        mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1) # (S, S)
        attn = torch.masked_fill(attn, mask, float("-inf")) # mask (S, S) broadcast -> (1, 1, S, S) -> (B, H, S, S)
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v # (B, H, S, D)

        # Output
        out = out.transpose(1, 2).reshape(B, S, self.head_dim * self.n_heads) # (B, S, D)
        out = self.out(out) # (B, S, input_dim)
        return out


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.w_up = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.w_down = nn.Linear(hidden_dim, input_dim)

    # x is (B, S, input_dim)
    def forward(self, x: torch.Tensor):
        return self.w_down(self.relu(self.w_up(x)))


class Block(nn.Module):
    def __init__(self, input_dim: int, attn_hidden_dim: int, mlp_hidden_dim: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(input_dim)
        self.attn = MultiHeadAttention(input_dim, attn_hidden_dim, n_heads)
        self.norm2 = nn.LayerNorm(input_dim)
        self.mlp = MLP(input_dim, mlp_hidden_dim)

    # x is (B, S, input_dim)
    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# Inherit from HF model class
class GPT(PreTrainedModel):
    config_class = ModelConfig
    _tied_weights_keys = {
        "lm_head.weight": "embedding.weight",
    }

    def __init__(
        self,
        config: ModelConfig
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
                config.n_heads
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
