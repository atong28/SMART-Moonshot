#!/usr/bin/env python3
import argparse
import os
import pickle
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

SPLITS = ("train", "val", "test")
MOD_DIRS = {
    "hsqc": "HSQC_NMR",
    "h_nmr": "H_NMR",
    "c_nmr": "C_NMR",
    "mass_spec": "MassSpec",
    "hmbc": "HMBC_NMR",
    "cosy": "COSY_NMR",
}
FRAGMENTS_DIR = "Fragments"


def _load_index(root: str) -> dict[int, Any]:
    with open(os.path.join(root, "index.pkl"), "rb") as f:
        return pickle.load(f)


def _iter_split_items(index: dict[int, Any], split: str):
    for idx, entry in index.items():
        if entry.get("split") == split:
            yield int(idx), entry


def _write_tensor_table(rows: list[tuple[int, list[float], list[int]]], output_path: str) -> None:
    table = pa.table(
        {
            "idx": [r[0] for r in rows],
            "data": [r[1] for r in rows],
            "shape": [r[2] for r in rows],
        },
        schema=pa.schema(
            [
                ("idx", pa.int64()),
                ("data", pa.list_(pa.float32())),
                ("shape", pa.list_(pa.int32())),
            ]
        ),
    )
    pq.write_table(table, output_path)


def _write_fragidx_table(rows: list[tuple[int, list[int]]], output_path: str) -> None:
    table = pa.table(
        {
            "idx": [r[0] for r in rows],
            "cols": [r[1] for r in rows],
        },
        schema=pa.schema(
            [
                ("idx", pa.int64()),
                ("cols", pa.list_(pa.int32())),
            ]
        ),
    )
    pq.write_table(table, output_path)


def _to_float32_1d(t: torch.Tensor) -> np.ndarray:
    arr = np.asarray(t.detach().cpu().numpy(), dtype=np.float32)
    return np.ascontiguousarray(arr.reshape(-1))


def convert_to_arrow(root: str, out_base: str, split: str | None, bitinfo_to_idx_path: str) -> None:
    index = _load_index(root)
    with open(bitinfo_to_idx_path, "rb") as f:
        bitinfo_to_idx: dict[Any, int] = pickle.load(f)
    get_col = bitinfo_to_idx.get

    splits = [split] if split else list(SPLITS)
    for current_split in splits:
        items = list(_iter_split_items(index, current_split))
        if not items:
            print(f"[warn] No items in split {current_split}, skipping")
            continue

        split_out = os.path.join(out_base, current_split)
        os.makedirs(split_out, exist_ok=True)
        print(f"\n=== Split: {current_split} ===")

        for key, mod_dir in MOD_DIRS.items():
            rows: list[tuple[int, list[float], list[int]]] = []
            for idx, entry in items:
                if not entry.get(f"has_{key}", False):
                    continue
                src = os.path.join(root, mod_dir, f"{idx}.pt")
                if not os.path.isfile(src):
                    continue
                tensor = torch.load(src, map_location="cpu", weights_only=True)
                arr_1d = _to_float32_1d(tensor)
                shape = [int(v) for v in tensor.shape]
                rows.append((idx, arr_1d.tolist(), shape))

            if not rows:
                continue

            out_path = os.path.join(split_out, f"{mod_dir}.parquet")
            _write_tensor_table(rows, out_path)
            print(f"[write] {out_path} rows={len(rows)}")

        frag_rows: list[tuple[int, list[int]]] = []
        for idx, _entry in items:
            frag_path = os.path.join(root, FRAGMENTS_DIR, f"{idx}.pt")
            if not os.path.isfile(frag_path):
                continue
            frag_list = torch.load(frag_path, map_location="cpu", weights_only=True)
            cols = sorted({get_col(bitinfo) for bitinfo in frag_list if get_col(bitinfo) is not None})
            frag_rows.append((idx, [int(v) for v in cols]))

        if frag_rows:
            frag_path = os.path.join(split_out, "FragIdx.parquet")
            _write_fragidx_table(frag_rows, frag_path)
            print(f"[write] {frag_path} rows={len(frag_rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert dataset tensors/fragments to Arrow parquet shards."
    )
    parser.add_argument("--root", required=True, help="DATASET_ROOT containing index.pkl and tensor folders")
    parser.add_argument(
        "--out",
        default=None,
        help="Output base for Arrow shards. Default: <root>/arrow",
    )
    parser.add_argument("--split", default=None, choices=SPLITS, help="Optional single split to convert")
    parser.add_argument(
        "--bitinfo-to-idx",
        required=True,
        help="Path to bitinfo_to_idx.pkl for building FragIdx.parquet.",
    )
    args = parser.parse_args()

    out_base = args.out or os.path.join(args.root, "arrow")
    convert_to_arrow(args.root, out_base, args.split, args.bitinfo_to_idx)
    print("\nAll done. Loaders expect Arrow tensors under <root>/arrow/<split>/*.parquet.")


if __name__ == "__main__":
    main()
