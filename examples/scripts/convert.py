#!/usr/bin/env python3
"""
Convert between Arrow-based dataset format and regular JSONL dataset format.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

SPLITS = ("train", "val", "test")
MOD_DIRS = {
    "hsqc": "HSQC_NMR",
    "h_nmr": "H_NMR",
    "c_nmr": "C_NMR",
    "mass_spec": "MassSpec",
    "fragidx": "FragIdx",
}

REGULAR_FILES = [
    "index.pkl",
    "retrieval.pkl",
    "metadata.json",
    "count_hashes_under_radius_6.pkl",
]
REGULAR_DIRS = ["RankingEntropy"]


def _copy_regular_data(input_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for filename in REGULAR_FILES:
        src = os.path.join(input_dir, filename)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output_dir, filename))
    for dirname in REGULAR_DIRS:
        src = os.path.join(input_dir, dirname)
        if os.path.isdir(src):
            dst = os.path.join(output_dir, dirname)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)


def convert_from_arrow(input_dir: str, output_dir: str) -> None:
    print(f"Converting from Arrow format: {input_dir} -> {output_dir}")
    _copy_regular_data(input_dir, output_dir)

    index_path = os.path.join(input_dir, "index.pkl")
    with open(index_path, "rb") as f:
        index = pickle.load(f)

    split_indices: Dict[str, List[int]] = {split: [] for split in SPLITS}
    for idx_key, entry in index.items():
        idx = int(idx_key)
        split = entry.get("split", "train")
        if split in split_indices:
            split_indices[split].append(idx)

    arrow_base = os.path.join(input_dir, "arrow")
    for split in SPLITS:
        indices = sorted(split_indices[split])
        if not indices:
            continue

        modality_rows: dict[str, dict[int, tuple[list[float], list[int]]]] = {}
        split_arrow_dir = os.path.join(arrow_base, split)
        for mod_key, mod_dir in MOD_DIRS.items():
            parquet_path = os.path.join(split_arrow_dir, f"{mod_dir}.parquet")
            if not os.path.isfile(parquet_path):
                continue
            table = pq.read_table(parquet_path)
            idx_values = table["idx"].to_pylist()
            if mod_key == "fragidx":
                cols_values = table["cols"].to_pylist()
                modality_rows[mod_key] = {int(idx): (cols, []) for idx, cols in zip(idx_values, cols_values)}
            else:
                data_values = table["data"].to_pylist()
                shape_values = table["shape"].to_pylist()
                modality_rows[mod_key] = {
                    int(idx): (data, shape)
                    for idx, data, shape in zip(idx_values, data_values, shape_values)
                }

        out_jsonl = os.path.join(output_dir, f"{split}.jsonl")
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for idx in tqdm(indices, desc=f"{split}"):
                entry = index[idx]
                json_entry: dict[str, Any] = {"idx": idx, "smiles": entry.get("smiles", ""), "split": split}
                for k, v in entry.items():
                    if k not in {"smiles", "split"}:
                        json_entry[k] = v

                for mod_key in MOD_DIRS:
                    rows = modality_rows.get(mod_key, {})
                    if idx not in rows:
                        continue
                    data, shape = rows[idx]
                    if mod_key == "fragidx":
                        json_entry["fragidx"] = [int(v) for v in data]
                        continue
                    if len(shape) == 2:
                        n, d = int(shape[0]), int(shape[1])
                        rebuilt = [data[i * d:(i + 1) * d] for i in range(n)]
                        if mod_key in {"h_nmr", "c_nmr"} and d == 1:
                            rebuilt = [row[0] for row in rebuilt]
                        json_entry[mod_key] = rebuilt
                    elif len(shape) == 1:
                        json_entry[mod_key] = data
                    else:
                        json_entry[mod_key] = data

                f.write(json.dumps(json_entry) + "\n")


def _flatten_with_shape(value: Any) -> tuple[list[float], list[int]]:
    if isinstance(value, list) and value and isinstance(value[0], list):
        shape = [len(value), len(value[0])]
        flat = [float(x) for row in value for x in row]
        return flat, shape
    if isinstance(value, list):
        shape = [len(value)]
        flat = [float(x) for x in value]
        return flat, shape
    return [], []


def convert_to_arrow(input_dir: str, output_dir: str) -> None:
    print(f"Converting to Arrow format: {input_dir} -> {output_dir}")
    _copy_regular_data(input_dir, output_dir)

    arrow_base = os.path.join(output_dir, "arrow")
    os.makedirs(arrow_base, exist_ok=True)

    for split in SPLITS:
        jsonl_path = os.path.join(input_dir, f"{split}.jsonl")
        if not os.path.isfile(jsonl_path):
            continue

        split_rows: dict[str, list[tuple[int, list[float], list[int]]]] = {
            "hsqc": [],
            "h_nmr": [],
            "c_nmr": [],
            "mass_spec": [],
        }
        frag_rows: list[tuple[int, list[int]]] = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"{split}"):
                if not line.strip():
                    continue
                row = json.loads(line)
                idx = int(row["idx"])
                for mod_key in split_rows:
                    if mod_key not in row or row[mod_key] in (None, []):
                        continue
                    flat, shape = _flatten_with_shape(row[mod_key])
                    split_rows[mod_key].append((idx, flat, shape))
                if "fragidx" in row and row["fragidx"] is not None:
                    frag_rows.append((idx, [int(v) for v in row["fragidx"]]))

        split_dir = os.path.join(arrow_base, split)
        os.makedirs(split_dir, exist_ok=True)
        for mod_key, mod_dir in MOD_DIRS.items():
            if mod_key == "fragidx":
                if not frag_rows:
                    continue
                table = {
                    "idx": [r[0] for r in frag_rows],
                    "cols": [r[1] for r in frag_rows],
                }
                pq.write_table(pa.table(table), os.path.join(split_dir, f"{mod_dir}.parquet"))
                continue

            rows = split_rows.get(mod_key, [])
            if not rows:
                continue
            table = {
                "idx": [r[0] for r in rows],
                "data": [r[1] for r in rows],
                "shape": [r[2] for r in rows],
            }
            pq.write_table(pa.table(table), os.path.join(split_dir, f"{mod_dir}.parquet"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert between Arrow and JSONL dataset formats")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--from-arrow", action="store_true", help="Convert from Arrow format to JSONL format")
    group.add_argument("--to-arrow", action="store_true", help="Convert from JSONL format to Arrow format")
    parser.add_argument("input_dir", help="Input dataset directory")
    parser.add_argument("output_dir", help="Output dataset directory")
    args = parser.parse_args()

    if args.from_arrow:
        convert_from_arrow(args.input_dir, args.output_dir)
    elif args.to_arrow:
        convert_to_arrow(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
