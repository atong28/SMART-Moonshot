#!/usr/bin/env python3
"""
Precompute DiffMS graph inputs (x, edge_index, edge_attr) and store them
as Arrow/Parquet shards, similar to the NMR storage layout.

Output layout (default):
    <root>/arrow/<split>/GraphX.parquet
    <root>/arrow/<split>/GraphEdgeIndex.parquet
    <root>/arrow/<split>/GraphEdgeAttr.parquet

Each Parquet file has schema compatible with ArrowTensorStore:
    - idx: int64
    - data: list<float32>
    - shape: list<int32>
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Any, Iterable, List, Tuple

from tqdm import tqdm
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from src.modules.core.const import DATASET_ROOT
from src.modules.data.inputs import GraphInputLoader

SPLITS: Tuple[str, ...] = ("train", "val", "test")


def _load_index(root: str) -> dict[int, Any]:
    index_path = os.path.join(root, "index.pkl")
    with open(index_path, "rb") as f:
        return pickle.load(f)


def _build_split_data_dict(index: dict[int, Any], split: str) -> dict[int, Any]:
    """
    Build a data_dict for the given split (dataset assumed prefiltered):
        - filter by entry['split'] == split
        - reindex to 0..N-1 in enumeration order
    """
    split_entries: List[Any] = []
    for _orig_idx, entry in tqdm(index.items(), desc=f'Building {split} data dict'):
        if entry.get("split") != split:
            continue
        smiles = entry.get("smiles")
        if not isinstance(smiles, str):
            continue
        split_entries.append(entry)
    return dict(enumerate(split_entries))


def _to_float32_1d(t: torch.Tensor) -> np.ndarray:
    arr = np.asarray(t.detach().cpu().numpy(), dtype=np.float32)
    return np.ascontiguousarray(arr.reshape(-1))


def _write_tensor_table(
    rows: Iterable[tuple[int, List[float], List[int]]],
    output_path: str,
) -> None:
    rows_list = list(rows)
    if not rows_list:
        return
    table = pa.table(
        {
            "idx": [r[0] for r in rows_list],
            "data": [r[1] for r in rows_list],
            "shape": [r[2] for r in rows_list],
        },
        schema=pa.schema(
            [
                ("idx", pa.int64()),
                ("data", pa.list_(pa.float32())),
                ("shape", pa.list_(pa.int32())),
            ]
        ),
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pq.write_table(table, output_path)


def _precompute_split(root: str, out_base: str, split: str) -> None:
    index = _load_index(root)
    data_dict = _build_split_data_dict(index, split)
    if not data_dict:
        print(f"[diffms_precompute] No entries found for split '{split}', skipping.")
        return

    print(f"[diffms_precompute] Split '{split}': {len(data_dict)} molecules after filtering.")

    graph_loader = GraphInputLoader(data_dict)  # compute from SMILES

    rows_x: List[tuple[int, List[float], List[int]]] = []
    rows_edge_index: List[tuple[int, List[float], List[int]]] = []
    rows_edge_attr: List[tuple[int, List[float], List[int]]] = []

    for idx in tqdm(range(len(data_dict)), desc=f'Precomputing {split} split'):
        try:
            x, edge_index, edge_attr, _smiles = graph_loader.load(idx)
        except Exception as e:
            print(f"[diffms_precompute] Error loading graph for idx {idx}: {data_dict[idx]['smiles']}")
            raise e

        x_arr = _to_float32_1d(x)
        ei_arr = _to_float32_1d(edge_index.to(torch.float32))
        ea_arr = _to_float32_1d(edge_attr)

        rows_x.append((idx, x_arr.tolist(), [int(v) for v in x.shape]))
        rows_edge_index.append((idx, ei_arr.tolist(), [int(v) for v in edge_index.shape]))
        rows_edge_attr.append((idx, ea_arr.tolist(), [int(v) for v in edge_attr.shape]))

    split_dir = os.path.join(out_base, split)
    print(f"[diffms_precompute] Writing Arrow shards to {split_dir}")

    _write_tensor_table(rows_x, os.path.join(split_dir, "GraphX.parquet"))
    _write_tensor_table(rows_edge_index, os.path.join(split_dir, "GraphEdgeIndex.parquet"))
    _write_tensor_table(rows_edge_attr, os.path.join(split_dir, "GraphEdgeAttr.parquet"))

    print(f"[diffms_precompute] Split '{split}' done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute DiffMS graph inputs into Arrow/Parquet shards."
    )
    parser.add_argument(
        "--root",
        default=str(DATASET_ROOT),
        help="Dataset root containing index.pkl (default: DATASET_ROOT from src.modules.core.const).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output base for Arrow shards (default: <root>/arrow).",
    )
    parser.add_argument(
        "--split",
        default=None,
        choices=SPLITS,
        help="Optional single split to convert (default: all).",
    )
    args = parser.parse_args()

    root = args.root
    out_base = args.out or os.path.join(root, "arrow")

    splits: Iterable[str]
    if args.split is not None:
        splits = (args.split,)
    else:
        splits = SPLITS

    print(f"[diffms_precompute] root={root}")
    print(f"[diffms_precompute] out_base={out_base}")

    for split in splits:
        _precompute_split(root, out_base, split)

    print("[diffms_precompute] All done.")


if __name__ == "__main__":
    main()

