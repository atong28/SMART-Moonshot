#!/usr/bin/env python
"""
Build FragIdx.lmdb for the NMRMind dataset once fragment bit information has
been generated.

Prerequisites (run beforehand):
  * convert_nmrmind_part_1.py  -> produces index.pkl + modality LMDB shards
  * entropy selection -> produces bitinfo_to_idx.pkl mapping BitInfo(tuple) -> column index

This script computes fragments directly from SMILES in index.pkl using the
specified radius, then maps them to column indices via bitinfo_to_idx.pkl.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import Dict, Iterable, List, Tuple

import lmdb
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.marina.data.fp_utils import count_circular_substructures, BitInfo

# TODO: Fix the hardcoded paths
DATASET_ROOT = "/data/nas-gpu/wang/atong/NMRMindDataset"
SPLITS = ("train", "val", "test")


def _load_index(root: str) -> Dict[int, Dict[str, object]]:
    path = os.path.join(root, "index.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"index.pkl not found at {path}. Run part 1 first.")
    with open(path, "rb") as fp:
        return pickle.load(fp)


def _iter_split_items(index: Dict[int, Dict[str, object]], split: str) -> Iterable[Tuple[int, Dict[str, object]]]:
    for idx, entry in index.items():
        if entry.get("split") == split:
            yield idx, entry


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


def build_fragidx_lmdb(
    root: str,
    out_base: str,
    bitinfo_to_idx_path: str,
    radius: int,
) -> None:
    index = _load_index(root)
    with open(bitinfo_to_idx_path, "rb") as fp:
        bitinfo_to_idx = pickle.load(fp)
    
    # Extract radius from bitinfo if not provided
    if radius is None:
        radius = _extract_radius_from_bitinfo(bitinfo_to_idx)
        print(f"[info] Extracted radius={radius} from bitinfo_to_idx")
    else:
        print(f"[info] Using provided radius={radius}")
    
    bit_lookup = bitinfo_to_idx.get

    for split in SPLITS:
        items = list(_iter_split_items(index, split))
        if not items:
            print(f"[info] No entries for split '{split}', skipping.")
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
                sorted({bit_lookup(b) for b in sample_frags.keys() if bit_lookup(b) is not None}),
                dtype=np.int32,
            )
        mapsize = _estimate_mapsize(len(items), max(len(_encode_array_raw(sample_cols)), 2048))

        print(f"[create] {lmdb_path} map_size≈{mapsize/1e9:.2f} GB (n={len(items)})")
        env = _open_env(lmdb_path, mapsize)

        count = 0
        batch: List[Tuple[int, bytes]] = []
        skipped = 0
        for i, (idx, entry) in enumerate(items, 1):
            smiles = entry.get("smiles")
            if not smiles:
                skipped += 1
                continue
            
            frags = count_circular_substructures(smiles, radius)
            cols = sorted({bit_lookup(b) for b in frags.keys() if bit_lookup(b) is not None})
            arr = np.asarray(cols, dtype=np.int32)
            buf = _encode_array_raw(arr)
            batch.append((idx, buf))

            if (i % 4096) == 0 or i == len(items):
                while True:
                    try:
                        with env.begin(write=True) as txn:
                            for key_idx, key_buf in batch:
                                txn.put(str(key_idx).encode("utf-8"), key_buf)
                        break
                    except lmdb.MapFullError:
                        env.set_mapsize(int(env.info()["map_size"] * 1.5))
                count += len(batch)
                print(f"  .. processed {i}/{len(items)} (total entries: {count}, skipped: {skipped})")
                batch.clear()

        env.sync()
        env.close()
        print(f"[done] {lmdb_path} entries={count}, skipped={skipped}")


def main():
    parser = argparse.ArgumentParser(description="Create FragIdx.lmdb for NMRMind dataset (part 2).")
    parser.add_argument("--root", default=DATASET_ROOT, help="Dataset root containing index.pkl")
    parser.add_argument(
        "--out",
        default=None,
        help="Base directory for LMDB shards (defaults to <root>/_lmdb).",
    )
    parser.add_argument(
        "--bitinfo-to-idx",
        required=True,
        help="Path to bitinfo_to_idx.pkl produced by entropy selection.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=None,
        help="Radius for circular substructure counting (default: extract from bitinfo_to_idx).",
    )
    args = parser.parse_args()

    out_base = args.out or os.path.join(args.root, "_lmdb")
    os.makedirs(out_base, exist_ok=True)

    build_fragidx_lmdb(
        root=args.root,
        out_base=out_base,
        bitinfo_to_idx_path=args.bitinfo_to_idx,
        radius=args.radius,
    )


if __name__ == "__main__":
    main()

