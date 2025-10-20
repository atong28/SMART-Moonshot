# spectre/lora/state_io.py
from __future__ import annotations
from typing import Dict, Any
import torch
import torch.nn as nn

def collect_lora_state(model: nn.Module) -> Dict[str, Any]:
    """
    Walks modules that expose .lora_state_dict() and collects them in a flat dict.
    Keys are module names; values are their LoRA state dicts.
    """
    out = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_state_dict"):
            sd = module.lora_state_dict()
            if sd:  # only store if non-empty
                out[name] = sd
    return out

@torch.no_grad()
def load_lora_state(model: nn.Module, lora_sd: Dict[str, Any]):
    for name, module in model.named_modules():
        if name in lora_sd and hasattr(module, "load_lora_state_dict"):
            module.load_lora_state_dict(lora_sd[name])

@torch.no_grad()
def clear_all_lora(model: nn.Module):
    for _, module in model.named_modules():
        if hasattr(module, "clear_lora"):
            module.clear_lora()
