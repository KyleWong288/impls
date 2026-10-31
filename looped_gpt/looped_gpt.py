import torch
import torch.nn as nn

from basic_gpt.basic_gpt import Block, ModelConfig
from transformers import PreTrainedModel


# Inherit from HF config base class, which requires defaults and kwargs plumbing
class ModelConfigLooped(ModelConfig):
    model_type = "looped_gpt"
    
    def __init__(
        self,
        n_loops: int = 2,
        use_outer_loop: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_loops = n_loops
        # Defaults to outer because it is better (Geiping 2025, Takase 2021)
        self.use_outer_loop = use_outer_loop


# Static amount of n_loops, mainly for param efficiency
class LoopedGPT(PreTrainedModel):
    config_class = ModelConfigLooped
    _tied_weights_keys = {
        "lm_head.weight": "embedding.weight",
    }

    def __init__(
        self,
        config: ModelConfigLooped
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
        self.norm = nn.RMSNorm(config.input_dim)
        self.lm_head = nn.Linear(config.input_dim, config.vocab_size, bias=False)
        # Tie weights for less params
        self.lm_head.weight = self.embedding.weight
        self.n_loops = config.n_loops
        self.use_outer_loop = config.use_outer_loop

    # x is (B, S)
    def forward(self, input_ids: torch.Tensor):
        x = self.embedding(input_ids) # (B, S, input_dim)

        if self.use_outer_loop:
            for _ in range(self.n_loops):
                for block in self.blocks:
                    x = block(x)
                # Normalize each time we loop through same block to prevent residual blowup (in this case, outer loop)
                x = self.norm(x)
        else:
            for block in self.blocks:
                for _ in range(self.n_loops):
                    x = block(x)
                    # Normalize each time we loop through same block to prevent residual blowup (in this case, inner loop)
                    x = self.norm(x)
        
        logits = self.lm_head(x) # (B, S, V)
        return logits
