# spectre/lora/adapter_bank.py
from __future__ import annotations
from typing import Dict, Any
import torch
from spectre.lora.state_io import load_lora_state, clear_all_lora

def combo_key(mods) -> str:
    return "+".join(sorted({m.lower() for m in mods}))

class AdapterBank:
    def __init__(self, device: str = "cpu"):
        self._store: Dict[str, Dict[str, Any]] = {}
        self.device = device

    def preload(self, key: str, path: str):
        sd = torch.load(path, map_location=self.device)
        self._store[key] = sd

    def has(self, key: str) -> bool:
        return key in self._store

    def activate(self, model, key: str):
        if key not in self._store:
            raise KeyError(f"Adapter not found for key={key}")
        load_lora_state(model, self._store[key])

    def clear(self, model):
        clear_all_lora(model)
