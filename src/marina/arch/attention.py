# spectre/arch/attention.py
from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionCore(nn.Module):
    """
    Custom multi-head attention with explicit Q/K/V/O projections.
    API mirrors nn.MultiheadAttention(batch_first=True) for our usage:
      forward(query, key, value, key_padding_mask=None) -> Tensor

    - Uses torch.nn.functional.scaled_dot_product_attention
    - Supports key_padding_mask (Boolean, True = pad/ignore)
    - Dropout is applied inside SDPA only during training
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = float(dropout)

        # Separate linear projections (weights compatible with PyTorch MHA splits)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    @torch.no_grad()
    def load_from_torch_mha(self, mha: nn.MultiheadAttention) -> None:
        """
        Convenience: copy weights from an nn.MultiheadAttention with the same dims.
        Assumes mha has batch_first=True and standard packed in_proj_weight/bias.
        """
        Wq, Wk, Wv = torch.chunk(mha.in_proj_weight, 3, dim=0)
        bq, bk, bv = torch.chunk(mha.in_proj_bias,   3, dim=0)
        self.q_proj.weight.copy_(Wq); self.q_proj.bias.copy_(bq)
        self.k_proj.weight.copy_(Wk); self.k_proj.bias.copy_(bk)
        self.v_proj.weight.copy_(Wv); self.v_proj.bias.copy_(bv)
        self.out_proj.weight.copy_(mha.out_proj.weight)
        self.out_proj.bias.copy_(mha.out_proj.bias)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, L, D) -> (B, H, L, Hd)
        """
        B, L, D = x.shape
        x = x.view(B, L, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(
        self,
        query: torch.Tensor,               # (B, Q, D)
        key: torch.Tensor,                 # (B, S, D)
        value: torch.Tensor,               # (B, S, D)
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, S) bool, True = pad
        attn_mask: Optional[torch.Tensor] = None,         # optional (Q,S) or broadcastable
        need_weights: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        Bq, Q, D = query.shape
        Bk, S, Dk = key.shape
        assert D == self.embed_dim and Dk == self.embed_dim
        assert Bq == Bk, "query/key batch sizes must match"

        # Projections
        q = self.q_proj(query)  # (B, Q, D)
        k = self.k_proj(key)    # (B, S, D)
        v = self.v_proj(value)  # (B, S, D)

        # Reshape to (B, H, L, Hd)
        q = self._shape(q)   # (B, H, Q, Hd)
        k = self._shape(k)   # (B, H, S, Hd)
        v = self._shape(v)   # (B, H, S, Hd)

        # Build attention mask for SDPA
        # torch SDPA expects attn_mask as either bool (True=mask) or additive float (-inf for mask)
        # We'll build an additive mask of shape (B, H, Q, S) for maximum compatibility.
        additive_mask = None
        if key_padding_mask is not None:
            # kpm: (B, S) -> (B, 1, 1, S)
            kpm = key_padding_mask.view(Bq, 1, 1, S).to(torch.bool)
            additive_mask = torch.zeros((Bq, self.num_heads, Q, S), device=query.device, dtype=q.dtype)
            additive_mask.masked_fill_(kpm, float("-inf"))

        if attn_mask is not None:
            # Broadcast/expand user mask to (B, H, Q, S) and add it
            # Accept bool (True=mask) or float (additive)
            if attn_mask.dtype == torch.bool:
                am = torch.zeros((Bq, self.num_heads, Q, S), device=query.device, dtype=q.dtype)
                am.masked_fill_(attn_mask, float("-inf"))
            else:
                am = attn_mask.to(dtype=q.dtype, device=query.device)
                if am.dim() == 2:  # (Q, S)
                    am = am.view(1, 1, Q, S).expand(Bq, self.num_heads, Q, S)
                elif am.dim() == 3:  # (B, Q, S)
                    am = am.view(Bq, 1, Q, S).expand(Bq, self.num_heads, Q, S)
            additive_mask = am if additive_mask is None else (additive_mask + am)

        # SDPA: (B, H, Q, Hd), (B, H, S, Hd) -> (B, H, Q, Hd)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=additive_mask,                  # additive mask or None
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )

        # Merge heads: (B, H, Q, Hd) -> (B, Q, D)
        y = y.permute(0, 2, 1, 3).contiguous().view(Bq, Q, D)
        y = self.out_proj(y)

        if not need_weights:
            return y

        # If weights are needed, compute them explicitly for parity with nn.MultiheadAttention
        # Softmax((q k^T) / sqrt(d) + mask) averaged over heads -> (B, Q, S)
        # Note: this is only for debugging/analysis; test parity uses need_weights=False.
        scale = 1.0 / math.sqrt(self.head_dim)
        logits = torch.einsum("bhiqd,bhjsd->bhqj", q, k) * scale  # (B, H, Q, S)
        if additive_mask is not None:
            logits = logits + additive_mask
        attn_weights = logits.softmax(dim=-1).mean(dim=1)  # average over heads
        return y, attn_weights
