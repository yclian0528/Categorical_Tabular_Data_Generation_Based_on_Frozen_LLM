import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.config_loader import load_config

class MMoELayer(nn.Module):
    """Multi-gate Mixture-of-Experts layer for task-specific features."""

    def __init__(
        self,
        input_dim,
        expert_dim,
        num_experts,
        num_tasks,
        hidden_dropout=None
    ):
        super().__init__()
        cfg = load_config()
        self.dropout_rate = hidden_dropout if hidden_dropout is not None else cfg.model.mmoe.dropout

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ) for _ in range(num_experts)
        ])

        self.gates = nn.ModuleList([
            nn.Linear(input_dim, num_experts) for _ in range(num_tasks)
        ])

    def forward(self, x, task_idx):
        """Return the gated expert output for one task."""
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)

        gate_logits = self.gates[task_idx](x)
        gate_weights = F.softmax(gate_logits, dim=-1)

        weighted_output = torch.bmm(gate_weights.unsqueeze(1), expert_outputs).squeeze(1)

        return weighted_output
