#!/usr/bin/env python
"""
Stream the raw NMRMind JSONL dataset directly into LMDB shards, skipping the
intermediate per-example ``*.pt`` files produced by ``nmrmind_dataset.py``.

For each split (train/val/test) we:
  * Parse the source JSON lines
  * Convert the available spectra into tensors
  * Append them straight into modality-specific LMDB databases
  * Build the same ``index.pkl`` metadata that downstream code expects

This script intentionally does **not** write ``FragIdx.lmdb``.  Run
``convert_nmrmind_part_2.py`` afterwards (once fragment bit info is prepared)
to populate that database.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import lmdb
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Dataset / modality constants
# ---------------------------------------------------------------------------

DATASET_ROOT = "/data/nas-gpu/wang/atong/NMRMindDataset"

SPLITS = ("train", "val", "test")
TRAIN_FILES = ["train0.json", "train1.json", "train2.json", "train3.json", "train4.json"]

# Map lower-case modality keys -> on-disk LMDB directory names (mirrors convert_to_lmdb.py)
MOD_DIRS = {
    "hsqc": "HSQC_NMR",
    "h_nmr": "H_NMR",
    "c_nmr": "C_NMR",
    "hmbc": "HMBC_NMR",
    "cosy": "COSY_NMR",
    # mass_spec omitted (dataset does not provide it)
}

# Source -> modality conversion helpers
COSY_TO_HSQC = "COSY"  # (H, C) pairs
HH_TO_COSY = "HH"      # (H, H) pairs


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def strip_c(s: str) -> float:
    return float(s.lstrip("C_"))


def strip_h(s: str) -> float:
    return float(s.lstrip("H_"))


def encode_tensor_raw(t: torch.Tensor) -> bytes:
    """Encode a tensor to bytes (dtype|ndim|shape| + raw data) for LMDB."""
    t = t.contiguous()
    arr = t.detach().cpu().numpy()
    arr = np.ascontiguousarray(arr)
    header = f"{arr.dtype.name}|{arr.ndim}|{','.join(map(str, arr.shape))}|".encode("ascii")
    return header + memoryview(arr)


@dataclass
class LMDBWriter:
    """Small helper that batches LMDB writes and auto-resizes the map."""

    path: str
    batch_size: int = 1024
    initial_map_size: int = 8 << 30  # 8 GiB

    def __post_init__(self):
        os.makedirs(self.path, exist_ok=True)
        self.env = lmdb.open(
            self.path,
            map_size=self.initial_map_size,
            subdir=True,
            lock=True,
            writemap=False,
            map_async=False,
            max_dbs=1,
        )
        self._batch: List[Tuple[int, bytes]] = []
        self.count = 0

    def add(self, idx: int, tensor: torch.Tensor):
        if tensor is None:
            return
        buf = encode_tensor_raw(tensor)
        self._batch.append((idx, buf))
        if len(self._batch) >= self.batch_size:
            self._flush()

    def _flush(self):
        if not self._batch:
            return
        while True:
            try:
                with self.env.begin(write=True) as txn:
                    for idx, buf in self._batch:
                        txn.put(str(idx).encode("utf-8"), buf)
                break
            except lmdb.MapFullError:
                current = self.env.info()["map_size"]
                self.env.set_mapsize(int(current * 1.5))
        self.count += len(self._batch)
        self._batch.clear()

    def close(self):
        self._flush()
        self.env.sync()
        self.env.close()


def build_split_file_list(split: str) -> List[str]:
    if split == "train":
        return TRAIN_FILES
    return [f"{split}.json"]


def ensure_split_dirs(out_base: str, split: str) -> Dict[str, LMDBWriter]:
    split_dir = os.path.join(out_base, split)
    os.makedirs(split_dir, exist_ok=True)
    writers: Dict[str, LMDBWriter] = {}
    for key, mod_dir in MOD_DIRS.items():
        lmdb_path = os.path.join(split_dir, f"{mod_dir}.lmdb")
        writers[key] = LMDBWriter(lmdb_path)
    return writers


def tensor_from_list(values: List[float]) -> Optional[torch.Tensor]:
    if not values:
        return None
    return torch.tensor(values, dtype=torch.float32)


def tensor_from_pairs(pairs: List[Tuple[float, float]]) -> Optional[torch.Tensor]:
    if not pairs:
        return None
    return torch.tensor(pairs, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Core conversion logic
# ---------------------------------------------------------------------------

def parse_and_convert(
    root: str,
    out_base: str,
    build_retrieval: bool = True,
) -> None:
    index: Dict[int, Dict[str, object]] = {}
    retrieval: Dict[int, Dict[str, str]] = {}

    split_writers: Dict[str, Dict[str, LMDBWriter]] = {
        split: ensure_split_dirs(out_base, split) for split in SPLITS
    }

    running_idx = 0
    seen_smiles: set[str] = set()
    skipped_dupes = 0
    for split in SPLITS:
        files = build_split_file_list(split)
        for file_name in files:
            path = os.path.join(root, file_name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing input file: {path}")
            with open(path, "r") as fp:
                for line in tqdm(fp, desc=f"{split}:{file_name}"):
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    smiles = data["smiles"]
                    if smiles in seen_smiles:
                        skipped_dupes += 1
                        continue
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        continue
                    seen_smiles.add(smiles)

                    entry_idx = running_idx
                    running_idx += 1

                    # Spectra tensors
                    tensors: Dict[str, Optional[torch.Tensor]] = {}

                    if data.get("13C_NMR"):
                        c_vals = [strip_c(v) for v in data["13C_NMR"]]
                        tensors["c_nmr"] = tensor_from_list(c_vals)
                    if data.get("1H_NMR"):
                        h_vals = [strip_h(pair[0]) for pair in data["1H_NMR"]]
                        tensors["h_nmr"] = tensor_from_list(h_vals)
                    if data.get(COSY_TO_HSQC):
                        hsqc_pairs = [
                            (strip_c(c_str), strip_h(h_str))
                            for h_str, c_str in data[COSY_TO_HSQC]
                        ]
                        tensors["hsqc"] = tensor_from_pairs(hsqc_pairs)
                    if data.get(HH_TO_COSY):
                        cosy_pairs = [
                            (strip_h(h1), strip_h(h2))
                            for h1, h2 in data[HH_TO_COSY]
                        ]
                        tensors["cosy"] = tensor_from_pairs(cosy_pairs)
                    if data.get("HMBC"):
                        hmbc_pairs = [
                            (strip_c(c_str), strip_h(h_str))
                            for h_str, c_str in data["HMBC"]
                        ]
                        tensors["hmbc"] = tensor_from_pairs(hmbc_pairs)

                    # Write tensors straight to LMDB
                    writers = split_writers[split]
                    for key, tensor in tensors.items():
                        if tensor is not None and key in writers:
                            writers[key].add(entry_idx, tensor)

                    entry = {
                        "smiles": smiles,
                        "has_hsqc": bool(data.get(COSY_TO_HSQC)),
                        "has_c_nmr": bool(data.get("13C_NMR")),
                        "has_h_nmr": bool(data.get("1H_NMR")),
                        "has_cosy": bool(data.get(HH_TO_COSY)),
                        "has_hmbc": bool(data.get("HMBC")),
                        "has_mass_spec": False,
                        "formula": data["molecular_formula"],
                        "split": split,
                        "has_mw": True,
                        "mw": rdMolDescriptors.CalcExactMolWt(mol),
                    }
                    index[entry_idx] = entry

                    if build_retrieval:
                        retrieval[entry_idx] = {"smiles": smiles}

    # Close LMDB writers
    for split_dict in split_writers.values():
        for writer in split_dict.values():
            writer.close()

    with open(os.path.join(root, "index.pkl"), "wb") as fp:
        pickle.dump(index, fp)

    if build_retrieval:
        with open(os.path.join(root, "retrieval.pkl"), "wb") as fp:
            pickle.dump(retrieval, fp)

    if skipped_dupes:
        print(f"[info] Skipped {skipped_dupes} duplicate SMILES entries.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Directly convert NMRMind JSON to LMDB shards (spectra only).")
    parser.add_argument("--root", default=DATASET_ROOT, help="Dataset root containing raw JSON files.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output base directory for LMDB shards. Defaults to <root>/_lmdb.",
    )
    parser.add_argument(
        "--no-retrieval",
        action="store_true",
        help="Skip writing retrieval.pkl (in case it is built separately).",
    )
    args = parser.parse_args()

    out_base = args.out or os.path.join(args.root, "_lmdb")
    os.makedirs(out_base, exist_ok=True)

    parse_and_convert(
        root=args.root,
        out_base=out_base,
        build_retrieval=not args.no_retrieval,
    )


if __name__ == "__main__":
    main()

