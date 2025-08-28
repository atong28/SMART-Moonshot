# spectre/lora/lora_layers.py
from __future__ import annotations
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """
    Identity-safe LoRA around a base Linear: y = base(x) + scale * (x @ B^T) @ A^T
    - A, B are zero-initialized so ΔW=0 at init (protects baseline).
    - Only A/B are considered "LoRA params".
    """
    def __init__(self, base_linear: nn.Linear, rank: int = 8, scale: float = 1.0):
        super().__init__()
        self.base = base_linear
        self.rank = int(rank)
        self.scale = float(scale)

        out_dim, in_dim = self.base.weight.shape
        if self.rank > 0:
            self.A = nn.Parameter(torch.zeros(out_dim, self.rank))
            self.B = nn.Parameter(torch.zeros(self.rank, in_dim))
        else:
            # keep attributes for typing; they won't be used
            self.register_parameter("A", None)
            self.register_parameter("B", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.rank <= 0:
            return y
        # Δ = (x @ B^T) @ A^T
        delta = (x @ self.B.t()) @ self.A.t()
        return y + self.scale * delta

    # --- utilities ---
    def lora_parameters(self):
        return [] if self.rank <= 0 else [self.A, self.B]

    def lora_state_dict(self) -> Dict[str, torch.Tensor]:
        if self.rank <= 0:
            return {}
        return {"A": self.A.detach().clone(), "B": self.B.detach().clone(), "scale": torch.tensor(self.scale), "rank": torch.tensor(self.rank)}

    @torch.no_grad()
    def load_lora_state_dict(self, sd: Dict[str, torch.Tensor]):
        if not sd:
            # treat as "clear"
            self.clear_lora()
            return
        r = int(sd.get("rank", torch.tensor(self.rank)).item())
        if r != self.rank:
            raise ValueError(f"Rank mismatch when loading LoRA: file rank={r}, module rank={self.rank}")
        self.A.copy_(sd["A"])
        self.B.copy_(sd["B"])
        if "scale" in sd:
            self.scale = float(sd["scale"].item())

    @torch.no_grad()
    def clear_lora(self):
        if self.rank > 0:
            self.A.zero_()
            self.B.zero_()

    @property
    def weight(self):
        # For API compatibility with modules that expect .weight (e.g., Transformer fast-path checks)
        return self.base.weight

    @property
    def bias(self):
        # Likewise for .bias
        return self.base.bias