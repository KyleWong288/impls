import torch
import torch.nn as nn

from basic_dataloader import BasicDataLoader
from basic_gpt import GPT, ModelConfig


class TrainConfig:
    batch_size: int
    lr: float
    clip_grad: float
    num_steps: int
    log_steps: int


def train(
    model: nn.Module,
    data_loader: BasicDataLoader,
    cfg: TrainConfig
):

    # Set up dataset iterator
    ds_iter = iter(data_loader)

    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    model.train()
    for i in range(cfg.num_steps):
        # inputs and targets are (B, S)
        inputs, targets = next(ds_iter)
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs) # (B, S, V)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), # (B*S, V)
            targets.reshape(-1), # (B*S,)
            ignore_index = data_loader.tokenizer.pad_token_id
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.clip_grad)
        optimizer.step()

        if i % cfg.log_steps == 0:
            print(f"Step {i}: loss={loss.item()}")


if __name__ == "__main__":
    train_cfg = TrainConfig(
        batch_size=64,
        lr=3e-4,
        clip_grad=1.0,
        num_steps=500,
        log_steps=10,
    )

    data_loader = BasicDataLoader(
        batch_size=train_cfg.batch_size,
        max_seq_len=256,
    )

    model_cfg = ModelConfig(
        vocab_size=len(data_loader.tokenizer),
        input_dim=128,
        attn_hidden_dim=128,
        mlp_hidden_dim=512,
        n_heads=4,
        n_blocks=4,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = GPT(model_cfg).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Training {parameter_count:,} parameters on {device}")

    train(model, data_loader, train_cfg)
    
