# spectre/lora/spectre_lora.py
from __future__ import annotations
import logging
from typing import Dict, Any, List, Iterable
import torch
import torch.nn as nn

# Base model & attention
from ..arch.model import SPECTRE
from ..arch.attention import MultiHeadAttentionCore

# LoRA components
from .attention_lora import LoRAttentionCore
from .lora_layers import LoRALinear
from .state_io import collect_lora_state, load_lora_state, clear_all_lora

def _replace_attn_with_lora(
    module: nn.Module,
    *,
    rank_qkv: int,
    rank_out: int,
    scale_qkv: float,
    scale_out: float,
    sites: List[str],
    prefix: str = "",
):
    """
    Recursively traverse `module` and replace any MultiHeadAttentionCore with LoRAttentionCore.
    Copies base weights into the LoRA-wrapped inner base linears.
    Records replaced module names into `sites`.
    """
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(child, MultiHeadAttentionCore):
            # Build LoRA variant with matching dims/dtype/device
            device = next(child.parameters()).device
            dtype  = next(child.parameters()).dtype

            lora_attn = LoRAttentionCore(
                embed_dim=child.embed_dim,
                num_heads=child.num_heads,
                dropout=child.dropout,
                bias=True,
                rank_qkv=rank_qkv,
                rank_out=rank_out,
                scale_qkv=scale_qkv,
                scale_out=scale_out,
            ).to(device=device, dtype=dtype)

            # Copy base weights into LoRA inner base linears
            with torch.no_grad():
                # q
                lora_attn.q_proj.base.weight.copy_(child.q_proj.weight)
                if child.q_proj.bias is not None:
                    lora_attn.q_proj.base.bias.copy_(child.q_proj.bias)
                # k
                lora_attn.k_proj.base.weight.copy_(child.k_proj.weight)
                if child.k_proj.bias is not None:
                    lora_attn.k_proj.base.bias.copy_(child.k_proj.bias)
                # v
                lora_attn.v_proj.base.weight.copy_(child.v_proj.weight)
                if child.v_proj.bias is not None:
                    lora_attn.v_proj.base.bias.copy_(child.v_proj.bias)
                # out
                lora_attn.out_proj.base.weight.copy_(child.out_proj.weight)
                if child.out_proj.bias is not None:
                    lora_attn.out_proj.base.bias.copy_(child.out_proj.bias)

            setattr(module, name, lora_attn)
            sites.append(full_name)

        else:
            _replace_attn_with_lora(
                child,
                rank_qkv=rank_qkv,
                rank_out=rank_out,
                scale_qkv=scale_qkv,
                scale_out=scale_out,
                sites=sites,
                prefix=full_name,
            )


def _maybe_wrap_fc_with_lora(
    model: nn.Module,
    *,
    rank_fc: int,
    scale_fc: float,
    sites: List[str],
):
    """Wrap model.fc with LoRA if present and rank_fc > 0."""
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear) and rank_fc > 0:
        fc = model.fc
        device = fc.weight.device
        dtype = fc.weight.dtype
        l_fc = LoRALinear(fc, rank=rank_fc, scale=scale_fc).to(device=device, dtype=dtype)
        model.fc = l_fc
        sites.append("fc")


def _iter_lora_modules(model: nn.Module):
    """
    Yield (name, module) for *leaf* LoRA modules only, to avoid recursion:
    - LoRALinear (wraps a base nn.Linear)
    - LoRAttentionCore (wraps Q/K/V/O projections)
    """
    for name, m in model.named_modules():
        # skip the root container itself
        if m is model:
            continue
        if isinstance(m, (LoRALinear, LoRAttentionCore)):
            yield name, m


def _lora_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    """Flatten LoRA parameters across the model."""
    params: List[nn.Parameter] = []
    for _, m in _iter_lora_modules(model):
        params.extend(m.lora_parameters())
    return params


def _freeze_base_enable_lora(model: nn.Module):
    """Freeze all params; re-enable grad only for LoRA params."""
    for p in model.parameters():
        p.requires_grad = False
    for _, m in _iter_lora_modules(model):
        for p in m.lora_parameters():
            p.requires_grad = True


class SPECTRELoRA(SPECTRE):
    """
    LoRA-enabled SPECTRE.
    - Same constructor signature as SPECTRE: SPECTRELoRA(args, fp_loader)
    - Reads LoRA knobs from args (with safe defaults):
        args.lora_rank_qkv  (int, default 8)
        args.lora_rank_out  (int, default 8)
        args.lora_rank_fc   (int, default 4)
        args.lora_scale_qkv (float, default 1.0)
        args.lora_scale_out (float, default 1.0)
        args.lora_scale_fc  (float, default 1.0)
    - Injects LoRA into all MultiHeadAttentionCore instances and optionally into final fc.
    - Provides helpers for adapter training & I/O.
    """

    def __init__(self, args, fp_loader):
        super().__init__(args, fp_loader)

        # Pull LoRA hyperparams from args with defaults
        rank_qkv  = args.lora_rank_qkv
        rank_out  = args.lora_rank_out
        rank_fc   = args.lora_rank_fc
        scale_qkv = args.lora_scale_qkv
        scale_out = args.lora_scale_out
        scale_fc  = args.lora_scale_fc

        # Inject LoRA
        self._lora_sites: List[str] = []
        _replace_attn_with_lora(
            self,
            rank_qkv=rank_qkv,
            rank_out=rank_out,
            scale_qkv=scale_qkv,
            scale_out=scale_out,
            sites=self._lora_sites,
            prefix="",
        )
        _maybe_wrap_fc_with_lora(
            self, rank_fc=rank_fc, scale_fc=scale_fc, sites=self._lora_sites
        )
        
        enable_self = args.lora_enable_self_attn
        if enable_self:
            raise NotImplementedError()

        try:
            logging.getLogger("lightning").info(
                "[LoRA] Enabled with sites: %s",
                ", ".join(self._lora_sites) if self._lora_sites else "(none)"
            )
        except Exception:
            pass

    # -------- Adapter utilities (instance methods) --------

    def lora_parameters(self) -> Iterable[nn.Parameter]:
        """Return an iterator over LoRA parameters."""
        return _lora_parameters(self)

    def freeze_base_enable_lora(self):
        """Freeze all base params; enable grad for LoRA params only."""
        _freeze_base_enable_lora(self)

    # -- LoRA state I/O --
    def save_adapter(self, path: str):
        """Save LoRA-only tensors to a small file."""
        sd = collect_lora_state(self)
        torch.save(sd, path)

    @torch.no_grad()
    def load_adapter(self, path: str, map_location: str | torch.device = "cpu"):
        """Load LoRA-only tensors from file and apply in-place."""
        sd = torch.load(path, map_location=map_location)
        load_lora_state(self, sd)

    @torch.no_grad()
    def clear_adapter(self):
        """Zero all LoRA deltas (return to pure base behavior)."""
        clear_all_lora(self)
