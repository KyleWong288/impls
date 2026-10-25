import torch
import torch.nn as nn

from basic_gpt.basic_gpt import MLP


class NaiveMoE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.router = nn.Linear(input_dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [MLP(input_dim, hidden_dim) for _ in range(num_experts)]
        ) 
        self.top_k = top_k
        self.num_experts = num_experts

    # x is (B, S, input_dim)
    def forward(self, x: torch.Tensor):
        batch_size, seq_len, input_dim = x.shape[0], x.shape[1], x.shape[2]
        total_tokens = batch_size * seq_len

        # Compute router probs
        router_scores = self.router(x) # (B, S, num_experts)
        router_probs = router_scores.softmax(dim=-1)

        # Compute selected experts and renormalize
        weights, indices = torch.topk(router_probs, self.top_k, dim=-1) # (B, S, top_k)
        weights = weights / weights.sum(dim=-1, keepdim=True) # (B, S, k)

        # Flatten
        weights = weights.reshape(total_tokens, self.top_k) # (B*S, k)
        indices = indices.reshape(total_tokens, self.top_k) # (B*S, k)
        x_flat = x.reshape(total_tokens, input_dim)         # (B*S, input_dim)
        out_flat = torch.zeros(total_tokens, input_dim, device=x.device, dtype=x.dtype)

        for expert_id, expert in enumerate(self.experts):
            mask = indices == expert_id # (B*S, k)
            # Gives the row id (which token) of each True, and col id (which choice) of each True
            token_ids, choice_ids = mask.nonzero(as_tuple=True) # (N,) where N is the # of True cells

            # Extract the assigned tokens, forward pass, and accumulate
            assigned_tokens = x_flat[token_ids]  # (N, input_dim)
            expert_out = expert(assigned_tokens) # (N, input_dim)
            weighted_out = expert_out * weights[token_ids, choice_ids, None] # (N, input_dim) * (N, 1)

            # index_add() avoids the row copying that advanced indexing with += would create
            out_flat = out_flat.index_add(0, token_ids, weighted_out)
        
        return out_flat.reshape(batch_size, seq_len, input_dim)
