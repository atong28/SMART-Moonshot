#!/usr/bin/env python
"""
Extend an existing LMDB-based dataset with additional NMRMind-style JSON data.

Workflow:
    1. Copy (or point to) an existing dataset root containing `_lmdb` shards.
    2. Run this script with `--root` pointing to the *new* combined dataset
       location and `--old-root` pointing to the source dataset to copy from.
    3. The script copies `_lmdb` from the old root if it doesn't already exist,
       loads the existing `index.pkl`/`retrieval.pkl`, and appends new entries
       both to the metadata files and to the LMDB shards in-place.
    4. It also records the list of newly assigned indices (default:
       `<root>/new_indices.pkl`) for part 2 to build FragIdx entries.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import lmdb
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_ROOT = "/data/nas-gpu/wang/atong/CombinedDataset"
SPLITS = ("train", "val", "test")
TRAIN_FILES = ["train0.json", "train1.json", "train2.json", "train3.json", "train4.json"]
MOD_DIRS = {
    "hsqc": "HSQC_NMR",
    "h_nmr": "H_NMR",
    "c_nmr": "C_NMR",
    "hmbc": "HMBC_NMR",
    "cosy": "COSY_NMR",
}
COSY_TO_HSQC = "COSY"
HH_TO_COSY = "HH"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_c(s: str) -> float:
    return float(s.lstrip("C_"))


def strip_h(s: str) -> float:
    return float(s.lstrip("H_"))


def encode_tensor_raw(t: torch.Tensor) -> bytes:
    t = t.contiguous()
    arr = np.ascontiguousarray(t.detach().cpu().numpy())
    header = f"{arr.dtype.name}|{arr.ndim}|{','.join(map(str, arr.shape))}|".encode("ascii")
    return header + memoryview(arr)


@dataclass
class LMDBWriter:
    """Append tensors to an existing LMDB shard, auto-resizing as needed."""

    path: str
    batch_size: int = 1024
    initial_map_size: int = 1 << 40  # ~1 TB

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

    def add(self, idx: int, tensor: torch.Tensor):
        if tensor is None:
            return
        self._batch.append((idx, encode_tensor_raw(tensor)))
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
        self._batch.clear()

    def close(self):
        self._flush()
        self.env.sync()
        self.env.close()


def build_split_file_list(split: str) -> List[str]:
    return TRAIN_FILES if split == "train" else [f"{split}.json"]


def ensure_split_writers(out_base: str, split: str) -> Dict[str, LMDBWriter]:
    split_dir = os.path.join(out_base, split)
    os.makedirs(split_dir, exist_ok=True)
    return {key: LMDBWriter(os.path.join(split_dir, f"{mod}.lmdb")) for key, mod in MOD_DIRS.items()}


def tensor_from_list(values: List[float]) -> Optional[torch.Tensor]:
    return torch.tensor(values, dtype=torch.float32) if values else None


def tensor_from_pairs(pairs: List[Tuple[float, float]]) -> Optional[torch.Tensor]:
    return torch.tensor(pairs, dtype=torch.float32) if pairs else None


def maybe_copy_tree(src: str, dst: str):
    if not os.path.exists(dst) or not os.listdir(dst):
        if not os.path.exists(src):
            raise FileNotFoundError(f"Source path for copy not found: {src}")
        shutil.copytree(src, dst, dirs_exist_ok=True)


def load_or_copy_pickle(path: str, fallback: str) -> Dict[int, Dict[str, object]]:
    if os.path.exists(path):
        with open(path, "rb") as fp:
            return pickle.load(fp)
    if os.path.exists(fallback):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        shutil.copy2(fallback, path)
        with open(path, "rb") as fp:
            return pickle.load(fp)
    return {}


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def parse_and_extend(
    root: str,
    out_base: str,
    old_root: str,
    build_retrieval: bool,
    new_indices_path: str,
) -> None:
    old_out = os.path.join(old_root, "_lmdb")
    maybe_copy_tree(old_out, out_base)

    index_path = os.path.join(root, "index.pkl")
    old_index_path = os.path.join(old_root, "index.pkl")
    index = load_or_copy_pickle(index_path, old_index_path)

    retrieval_path = os.path.join(root, "retrieval.pkl")
    old_retrieval_path = os.path.join(old_root, "retrieval.pkl")
    retrieval = load_or_copy_pickle(retrieval_path, old_retrieval_path) if build_retrieval else {}

    next_idx = max(index.keys(), default=-1) + 1
    seen_smiles: set[str] = {entry["smiles"] for entry in index.values()}
    skipped_dupes = 0
    new_indices: List[int] = []

    split_writers: Dict[str, Dict[str, LMDBWriter]] = {
        split: ensure_split_writers(out_base, split) for split in SPLITS
    }

    for split in SPLITS:
        files = build_split_file_list(split)
        for file_name in files:
            path = os.path.join(root, file_name)
            if not os.path.exists(path):
                continue
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

                    entry_idx = next_idx
                    next_idx += 1
                    new_indices.append(entry_idx)

                    tensors: Dict[str, Optional[torch.Tensor]] = {}
                    if data.get("13C_NMR"):
                        tensors["c_nmr"] = tensor_from_list([strip_c(v) for v in data["13C_NMR"]])
                    if data.get("1H_NMR"):
                        tensors["h_nmr"] = tensor_from_list([strip_h(pair[0]) for pair in data["1H_NMR"]])
                    if data.get(COSY_TO_HSQC):
                        tensors["hsqc"] = tensor_from_pairs([
                            (strip_c(c_str), strip_h(h_str)) for h_str, c_str in data[COSY_TO_HSQC]
                        ])
                    if data.get(HH_TO_COSY):
                        tensors["cosy"] = tensor_from_pairs([
                            (strip_h(h1), strip_h(h2)) for h1, h2 in data[HH_TO_COSY]
                        ])
                    if data.get("HMBC"):
                        tensors["hmbc"] = tensor_from_pairs([
                            (strip_c(c_str), strip_h(h_str)) for h_str, c_str in data["HMBC"]
                        ])

                    for key, tensor in tensors.items():
                        if tensor is not None:
                            split_writers[split][key].add(entry_idx, tensor)

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

    for writer_dict in split_writers.values():
        for writer in writer_dict.values():
            writer.close()

    with open(index_path, "wb") as fp:
        pickle.dump(index, fp)

    if build_retrieval:
        with open(retrieval_path, "wb") as fp:
            pickle.dump(retrieval, fp)

    if new_indices:
        with open(new_indices_path, "wb") as fp:
            pickle.dump(new_indices, fp)
    if skipped_dupes:
        print(f"[info] Skipped {skipped_dupes} duplicate SMILES entries (already present).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extend existing LMDB shards with additional NMRMind data.")
    parser.add_argument("--root", default=DATASET_ROOT, help="Combined dataset root containing raw JSON files.")
    parser.add_argument("--old-root", required=True, help="Existing dataset root to copy `_lmdb` (and metadata) from.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output base for LMDB shards (defaults to <root>/_lmdb).",
    )
    parser.add_argument(
        "--no-retrieval",
        action="store_true",
        help="Skip updating retrieval.pkl.",
    )
    parser.add_argument(
        "--new-indices-path",
        default=None,
        help="Where to save the list of newly added indices (defaults to <root>/new_indices.pkl).",
    )
    args = parser.parse_args()

    out_base = args.out or os.path.join(args.root, "_lmdb")
    new_indices_path = args.new_indices_path or os.path.join(args.root, "new_indices.pkl")
    os.makedirs(out_base, exist_ok=True)

    parse_and_extend(
        root=args.root,
        out_base=out_base,
        old_root=args.old_root,
        build_retrieval=not args.no_retrieval,
        new_indices_path=new_indices_path,
    )


if __name__ == "__main__":
    main()

