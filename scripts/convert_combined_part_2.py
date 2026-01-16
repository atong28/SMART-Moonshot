#!/usr/bin/env python
"""
Continue writing FragIdx.lmdb for a combined dataset.

Assumes `convert_combined_part_1.py` has already appended new entries and
produced `<root>/new_indices.pkl` listing the indices that still need fragment
records. This script will:
    * Load the combined index
    * Restrict to the provided set of new indices (or all entries if file absent)
    * Compute fragments directly from SMILES in index.pkl
    * Append rows to the existing FragIdx.lmdb shards in `_lmdb/<split>/FragIdx.lmdb`
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import lmdb
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.marina.data.fp_utils import count_circular_substructures, BitInfo

# TODO: Fix the hardcoded paths
DATASET_ROOT = "/data/nas-gpu/wang/atong/CombinedDataset"
SPLITS = ("train", "val", "test")


def _load_index(root: str) -> Dict[int, Dict[str, object]]:
    path = os.path.join(root, "index.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing index file: {path}")
    with open(path, "rb") as fp:
        return pickle.load(fp)


def _iter_split_items(index: Dict[int, Dict[str, object]], split: str):
    for idx, entry in index.items():
        if entry.get("split") == split:
            yield idx, entry


def _load_new_indices(path: Optional[str]) -> Optional[set[int]]:
    if path and os.path.exists(path):
        with open(path, "rb") as fp:
            return set(pickle.load(fp))
    return None


def _encode_array_raw(arr: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(arr)
    header = f"{arr.dtype.name}|{arr.ndim}|{','.join(map(str, arr.shape))}|".encode("ascii")
    return header + memoryview(arr)


def _estimate_mapsize(num_items: int, avg_bytes: int) -> int:
    return int(max(1, num_items) * max(1, avg_bytes) * 1.3) + (256 << 20)


def _open_env(path: str, mapsize: int):
    os.makedirs(path, exist_ok=True)
    return lmdb.open(
        path,
        map_size=mapsize,
        subdir=True,
        lock=True,
        writemap=False,
        map_async=False,
        max_dbs=1,
    )


def _extract_radius_from_bitinfo(bitinfo_to_idx: Dict[BitInfo, int]) -> int:
    """Extract radius from the first BitInfo key in the mapping."""
    if not bitinfo_to_idx:
        raise ValueError("bitinfo_to_idx is empty")
    first_bitinfo = next(iter(bitinfo_to_idx.keys()))
    if not isinstance(first_bitinfo, tuple) or len(first_bitinfo) != 4:
        raise ValueError(f"Invalid BitInfo format: {first_bitinfo}")
    return int(first_bitinfo[3])  # radius is the 4th element


def build_fragidx(
    root: str,
    out_base: str,
    bitinfo_to_idx_path: str,
    new_indices_path: Optional[str],
    radius: Optional[int],
) -> None:
    index = _load_index(root)
    os.makedirs(out_base, exist_ok=True)

    with open(bitinfo_to_idx_path, "rb") as fp:
        bitinfo_to_idx = pickle.load(fp)
    
    # Extract radius from bitinfo if not provided
    if radius is None:
        radius = _extract_radius_from_bitinfo(bitinfo_to_idx)
        print(f"[info] Extracted radius={radius} from bitinfo_to_idx")
    else:
        print(f"[info] Using provided radius={radius}")
    
    lookup = bitinfo_to_idx.get

    new_indices = _load_new_indices(new_indices_path)
    if new_indices:
        print(f"[info] Restricting FragIdx build to {len(new_indices)} new entries.")

    for split in SPLITS:
        items = [
            (idx, entry)
            for idx, entry in _iter_split_items(index, split)
            if (new_indices is None or idx in new_indices)
        ]

        if not items:
            print(f"[info] No entries to add for split '{split}'.")
            continue

        split_dir = os.path.join(out_base, split)
        os.makedirs(split_dir, exist_ok=True)
        lmdb_path = os.path.join(split_dir, "FragIdx.lmdb")

        # Estimate sample size from first entry
        sample_idx, sample_entry = items[0]
        sample_smiles = sample_entry.get("smiles")
        if not sample_smiles:
            print(f"[warn] Entry {sample_idx} missing SMILES, skipping sample estimation")
            sample_cols = np.array([], dtype=np.int32)
        else:
            sample_frags = count_circular_substructures(sample_smiles, radius)
            sample_cols = np.asarray(
                sorted({lookup(b) for b in sample_frags.keys() if lookup(b) is not None}),
                dtype=np.int32,
            )
        sample_bytes = max(len(_encode_array_raw(sample_cols)), 2048)
        mapsize = _estimate_mapsize(len(items), sample_bytes)

        print(f"[open] {lmdb_path} (target entries={len(items)}, map_size≈{mapsize/1e9:.2f} GB)")
        env = _open_env(lmdb_path, mapsize)

        count = 0
        batch: List[Tuple[int, bytes]] = []
        skipped = 0
        total = len(items)
        for i, (idx, entry) in enumerate(items, 1):
            smiles = entry.get("smiles")
            if not smiles:
                skipped += 1
                continue
            
            frags = count_circular_substructures(smiles, radius)
            cols = sorted({lookup(b) for b in frags.keys() if lookup(b) is not None})
            arr = np.asarray(cols, dtype=np.int32)
            buf = _encode_array_raw(arr)
            batch.append((idx, buf))

            if (i % 4096) == 0 or i == total:
                while True:
                    try:
                        with env.begin(write=True) as txn:
                            for key_idx, key_buf in batch:
                                txn.put(str(key_idx).encode("utf-8"), key_buf)
                        break
                    except lmdb.MapFullError:
                        env.set_mapsize(int(env.info()["map_size"] * 1.5))
                count += len(batch)
                print(f"  .. processed {i}/{total} (total entries={count}, skipped={skipped})")
                batch.clear()

        env.sync()
        env.close()
        print(f"[done] {lmdb_path} appended={count}, skipped={skipped}")


def main():
    parser = argparse.ArgumentParser(description="Extend FragIdx.lmdb for a combined dataset.")
    parser.add_argument("--root", default=DATASET_ROOT, help="Combined dataset root.")
    parser.add_argument(
        "--out",
        default=None,
        help="Base directory for LMDB shards (defaults to <root>/_lmdb).",
    )
    parser.add_argument(
        "--bitinfo-to-idx",
        required=True,
        help="Path to bitinfo_to_idx.pkl (matches combined dataset).",
    )
    parser.add_argument(
        "--new-indices-path",
        default=None,
        help="Optional path to the pickle produced by convert_combined_part_1.py.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=None,
        help="Radius for circular substructure counting (default: extract from bitinfo_to_idx).",
    )
    args = parser.parse_args()

    out_base = args.out or os.path.join(args.root, "_lmdb")
    build_fragidx(
        root=args.root,
        out_base=out_base,
        bitinfo_to_idx_path=args.bitinfo_to_idx,
        new_indices_path=args.new_indices_path or os.path.join(args.root, "new_indices.pkl"),
        radius=args.radius,
    )


if __name__ == "__main__":
    main()

