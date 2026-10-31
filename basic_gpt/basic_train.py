import os
import time
import torch
import torch.nn as nn

from basic_dataloader import BasicDataLoader
from basic_gpt import GPT, ModelConfig
from dataclasses import dataclass
from transformers import PreTrainedModel


@dataclass
class TrainConfig:
    batch_size: int
    lr: float
    clip_grad: float
    train_steps: int
    log_steps: int
    save_steps: int
    save_dir: str


def train(
    model: PreTrainedModel,
    data_loader: BasicDataLoader,
    cfg: TrainConfig
):

    # Set up dataset iterator
    ds_iter = iter(data_loader)

    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # Set up timing metrics
    start_time = time.perf_counter()

    model.train()
    for step in range(cfg.train_steps):

        # inputs and targets are (B, S)
        inputs, targets = next(ds_iter)
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward and backward
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

        # Save checkpoint
        if (step + 1) % cfg.save_steps == 0:
            print(f"Saving checkpoint step {step}: loss={loss.item()}, grad_norm={grad_norm.item()}")
            checkpoint_path = os.path.join(cfg.save_dir, f"checkpoint-{step + 1}")
            model.save_pretrained(checkpoint_path, safe_serialization=True)
            data_loader.tokenizer.save_pretrained(checkpoint_path)

        # Log checkpoint
        elif step % cfg.log_steps == 0:
            # item() implicitly synchronizes device because it copies to CPU, which makes our timing accurate
            loss_value = loss.item()
            grad_norm_value = grad_norm.item()
            mean_step_time = (time.perf_counter() - start_time) / (step + 1)
            print(f"Logging step {step}: loss={loss_value:.4f}, grad_norm={grad_norm_value:.4f}, mean_step_time={mean_step_time:.4f}s")


# ~20M params, ~400M tokens
if __name__ == "__main__":
    train_cfg = TrainConfig(
        batch_size=256,
        lr=3e-4,
        clip_grad=1.0,
        train_steps=6000,
        log_steps=10,
        save_steps=500,
        save_dir="./checkpoints/"
    )

    data_loader = BasicDataLoader(
        batch_size=train_cfg.batch_size,
        max_seq_len=256,
    )

    model_cfg = ModelConfig(
        vocab_size=len(data_loader.tokenizer),
        input_dim=256,
        attn_hidden_dim=256,
        mlp_hidden_dim=1024,
        n_heads=4,
        n_blocks=8,
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
