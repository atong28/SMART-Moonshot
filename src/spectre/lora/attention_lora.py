# spectre/lora/attention_lora.py
from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..arch.attention import MultiHeadAttentionCore
from .lora_layers import LoRALinear

class LoRAttentionCore(MultiHeadAttentionCore):
    """
    Same API as MultiHeadAttentionCore, but Q/K/V/O projections are LoRA-wrapped.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True,
                 rank_qkv: int = 8, rank_out: int = 8, scale_qkv: float = 1.0, scale_out: float = 1.0):
        super().__init__(embed_dim, num_heads, dropout=dropout, bias=bias)
        # Wrap the base linears with LoRA
        self.q_proj = LoRALinear(self.q_proj, rank=rank_qkv, scale=scale_qkv)
        self.k_proj = LoRALinear(self.k_proj, rank=rank_qkv, scale=scale_qkv)
        self.v_proj = LoRALinear(self.v_proj, rank=rank_qkv, scale=scale_qkv)
        self.out_proj = LoRALinear(self.out_proj, rank=rank_out, scale=scale_out)

    # --- LoRA state helpers across all four sites ---
    def lora_state_dict(self) -> Dict[str, Any]:
        return {
            "q": self.q_proj.lora_state_dict(),
            "k": self.k_proj.lora_state_dict(),
            "v": self.v_proj.lora_state_dict(),
            "o": self.out_proj.lora_state_dict(),
        }

    @torch.no_grad()
    def load_lora_state_dict(self, sd: Dict[str, Any]):
        self.q_proj.load_lora_state_dict(sd.get("q", {}))
        self.k_proj.load_lora_state_dict(sd.get("k", {}))
        self.v_proj.load_lora_state_dict(sd.get("v", {}))
        self.out_proj.load_lora_state_dict(sd.get("o", {}))

    @torch.no_grad()
    def clear_lora(self):
        self.q_proj.clear_lora()
        self.k_proj.clear_lora()
        self.v_proj.clear_lora()
        self.out_proj.clear_lora()

    def lora_parameters(self):
        params = []
        for mod in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            params.extend(mod.lora_parameters())
        return params

    @torch.no_grad()
    def load_from_torch_mha(self, mha: nn.MultiheadAttention) -> None:
        """
        Copy weights from a torch.nn.MultiheadAttention (batch_first=True).
        Writes into *base* weights under the LoRA wrappers.
        """
        Wq, Wk, Wv = torch.chunk(mha.in_proj_weight, 3, dim=0)
        bq, bk, bv = torch.chunk(mha.in_proj_bias,   3, dim=0)
        # Copy into inner base linears
        self.q_proj.base.weight.copy_(Wq); self.q_proj.base.bias.copy_(bq)
        self.k_proj.base.weight.copy_(Wk); self.k_proj.base.bias.copy_(bk)
        self.v_proj.base.weight.copy_(Wv); self.v_proj.base.bias.copy_(bv)
        self.out_proj.base.weight.copy_(mha.out_proj.weight)
        self.out_proj.base.bias.copy_(mha.out_proj.bias)

