#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import time
import argparse
import multiprocessing as mp
from typing import Dict, List, Optional, Set, Tuple, DefaultDict
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from rdkit import Chem

# Headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.marina.core.const import DATASET_ROOT
from src.marina.data.fp_loader import EntropyFPLoader
from src.marina.data.fp_utils import (
    BitInfo,           # (bit_id, atom_symbol, fragment_smiles, radius)
    load_smiles_index,
    get_bitinfos,      # (atom_to_bit_infos, all_bit_infos)
)

# ---------------------------
# Globals for worker processes
# ---------------------------
G_RADIUS: Optional[int] = None
G_MIN_RADIUS: Optional[int] = None
G_SELECTED: Optional[Set[BitInfo]] = None
G_MODE: Optional[str] = None  # "center" or "env"


def _init_worker(radius: int, min_radius: int, selected: Set[BitInfo], mode: str) -> None:
    """Initializer for multiprocessing workers."""
    global G_RADIUS, G_MIN_RADIUS, G_SELECTED, G_MODE
    G_RADIUS = int(radius)
    G_MIN_RADIUS = int(min_radius)
    G_SELECTED = set(selected)  # local copy per worker
    G_MODE = mode


def _coverage_for_row(args: Tuple[int, str]) -> Tuple[int, float, bool]:
    """
    Compute atom coverage for a single SMILES with binary-FP semantics:
      - Each unique selected BitInfo contributes at most once per molecule.
      - We union atoms per BitInfo either as:
          * center: only the center atom
          * env:    all atoms in the radius-N environment around the center
    Returns (row_idx, coverage in [0,1], is_zero_coverage).
    """
    assert G_RADIUS is not None and G_MIN_RADIUS is not None and G_SELECTED is not None and G_MODE is not None
    row_idx, smi = args

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return row_idx, 0.0, True
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return row_idx, 0.0, True

    atom_to_bitinfos, _ = get_bitinfos(smi, G_RADIUS)
    if atom_to_bitinfos is None:
        return row_idx, 0.0, True

    # Build: BitInfo -> set(atom_indices) for *this* molecule, filtering by min-radius
    bit_to_atoms: DefaultDict[BitInfo, Set[int]] = defaultdict(set)
    if G_MODE == "center":
        # Only center atom counts
        for center_idx in range(num_atoms):
            for b in atom_to_bitinfos.get(center_idx, ()):
                r = b[3]
                if r >= G_MIN_RADIUS:
                    bit_to_atoms[b].add(center_idx)
    else:
        # Full environment coverage
        for center_idx in range(num_atoms):
            for b in atom_to_bitinfos.get(center_idx, ()):
                r = b[3]
                if r >= G_MIN_RADIUS:
                    env_bond_ids = Chem.FindAtomEnvironmentOfRadiusN(mol, r, center_idx)
                    env_atom_ids = {center_idx}
                    for bd in env_bond_ids:
                        bond = mol.GetBondWithIdx(bd)
                        env_atom_ids.add(bond.GetBeginAtomIdx())
                        env_atom_ids.add(bond.GetEndAtomIdx())
                    bit_to_atoms[b].update(env_atom_ids)

    # Intersect with selected vocabulary (binary presence semantics: each BitInfo used once)
    selected_present_bits = (set(bit_to_atoms.keys()) & G_SELECTED)

    covered_atoms = np.zeros(num_atoms, dtype=bool)
    for b in selected_present_bits:
        idxs = bit_to_atoms[b]
        if idxs:
            covered_atoms[list(idxs)] = True

    covered = int(covered_atoms.sum())
    cov = float(covered) / float(num_atoms) if num_atoms else 0.0
    is_zero = (covered == 0)
    return row_idx, cov, is_zero


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute atom-level coverage by selected fingerprint bits (binary semantics).")
    p.add_argument("--retrieval", type=str, default=os.path.join(DATASET_ROOT, "retrieval.pkl"),
                   help="Path to retrieval index (.pkl/.json) with smiles.")
    p.add_argument("--dataset-root", type=str, default=DATASET_ROOT,
                   help="Dataset root (base for outputs).")
    p.add_argument("--radius", type=int, default=6, help="Max Morgan radius considered (<=R).")
    p.add_argument("--min-radius", type=int, default=1,
                   help="Minimum radius to count for coverage (exclude radius < min-radius).")
    p.add_argument("--out-dim", default=16384,
                   help="Entropy fingerprint size (int) or 'inf' to select all).")
    p.add_argument("--fp-type", type=str, default="RankingEntropy",
                   help="Name used when saving artifacts.")
    p.add_argument("--num-procs", type=int, default=0,
                   help="0=auto (all CPUs), 1=serial, else explicit number.")
    p.add_argument("--chunk-size", type=int, default=256,
                   help="Chunksize for imap_unordered.")
    p.add_argument("--hist-bins", type=int, default=50,
                   help="Number of bins for the coverage histogram.")
    p.add_argument("--viz-dir", type=str, default="visualizations",
                   help="Directory (relative or absolute) to write visualization artifacts.")
    p.add_argument("--coverage-mode", type=str, choices=("center", "env"), default="env",
                   help="How to mark covered atoms per BitInfo: 'center' = only center atom; 'env' = full radius-N environment.")
    return p.parse_args()


def _make_names(fp_type: str, radius: int, k_str: str, min_r: int, mode: str) -> Dict[str, str]:
    base = f"{fp_type}_R{radius}_K{k_str}_minR{min_r}_{mode}"
    return {
        "summary_json": f"coverage_{base}.json",
        "hist_json":    f"coverage_hist_{base}.json",
        "hist_csv":     f"coverage_hist_{base}.csv",
        "hist_png":     f"coverage_hist_{base}.png",
        "zero_txt":     f"zero_coverage_{base}.txt",
    }


def main() -> None:
    args = parse_args()
    t0 = time.time()

    # Normalize out_dim
    out_dim = "inf" if str(args.out_dim).lower() == "inf" else int(args.out_dim)

    if args.min_radius < 0 or args.min_radius > args.radius:
        raise ValueError(f"--min-radius must be in [0, {args.radius}] (got {args.min_radius}).")

    # 1) Load/prepare selected vocabulary
    loader = EntropyFPLoader(dataset_root=args.dataset_root, retrieval_path=args.retrieval)
    loader.setup(out_dim, args.radius, retrieval_path=args.retrieval, num_procs=max(0, int(args.num_procs)))
    if not loader.bitinfo_to_fp_index_map:
        raise RuntimeError("No selected features; did setup() run?")

    # 2) Filter selected features by min_radius (drop r < min_radius)
    selected_all: Set[BitInfo] = set(loader.bitinfo_to_fp_index_map.keys())
    selected_filtered: Set[BitInfo] = {b for b in selected_all if b[3] >= args.min_radius}
    effective_k = len(selected_filtered)
    if effective_k == 0:
        raise RuntimeError(
            f"All selected features were below min-radius={args.min_radius}. "
            "Re-run selection with a larger K or smaller min-radius."
        )
    drop_ct = len(selected_all) - effective_k
    if drop_ct > 0:
        print(f"[count_coverage] Filtered out {drop_ct} bits with radius < {args.min_radius} "
              f"(kept {effective_k} / selected {len(selected_all)}).")

    # 3) Load retrieval set; keep stable order and original retrieval indices
    smiles_map: Dict[int, str] = load_smiles_index(args.retrieval)
    ordered_items: List[Tuple[int, str]] = sorted(smiles_map.items(), key=lambda kv: int(kv[0]))
    ordered_rows: List[Tuple[int, str]] = [(i, smi) for i, (_, smi) in enumerate(ordered_items)]
    ordered_keys: List[int] = [idx for idx, _ in ordered_items]
    ordered_smiles: List[str] = [smi for _, smi in ordered_items]
    num_rows = len(ordered_rows)
    if num_rows == 0:
        raise RuntimeError("Retrieval set is empty.")

    # 4) Coverage computation (multiprocess if requested)
    procs = (mp.cpu_count() if int(args.num_procs) == 0 else max(1, int(args.num_procs)))
    coverages: List[float] = [0.0] * num_rows
    zero_flags: List[bool] = [False] * num_rows

    if procs == 1:
        _init_worker(args.radius, args.min_radius, selected_filtered, args.coverage_mode)
        for row_idx, smi in tqdm(ordered_rows, total=num_rows, desc=f"Computing coverage ({args.coverage_mode}, serial)"):
            i, cov, is_zero = _coverage_for_row((row_idx, smi))
            coverages[i] = cov
            zero_flags[i] = is_zero
    else:
        with mp.Pool(
            processes=procs,
            initializer=_init_worker,
            initargs=(args.radius, args.min_radius, selected_filtered, args.coverage_mode),
        ) as pool:
            for i, cov, is_zero in tqdm(
                pool.imap_unordered(_coverage_for_row, ordered_rows, chunksize=max(1, int(args.chunk_size))),
                total=num_rows,
                desc=f"Computing coverage ({args.coverage_mode}, mp x{procs})",
            ):
                coverages[i] = cov
                zero_flags[i] = is_zero

    # 5) Aggregate stats + zero-coverage diagnostics
    cov_array = np.asarray(coverages, dtype=np.float64)
    mean_cov = float(cov_array.mean()) if cov_array.size else 0.0
    min_cov = float(cov_array.min()) if cov_array.size else 0.0
    max_cov = float(cov_array.max()) if cov_array.size else 0.0

    zero_indices = [i for i, flag in enumerate(zero_flags) if flag]
    zero_count = len(zero_indices)
    first_zero_idx = zero_indices[0] if zero_indices else None
    first_zero_smiles = ordered_smiles[first_zero_idx] if first_zero_idx is not None else None
    first_zero_retrieval_index = ordered_keys[first_zero_idx] if first_zero_idx is not None else None

    # 6) Prepare output paths
    k_str = "inf" if out_dim == "inf" else str(out_dim)
    names = _make_names(args.fp_type, int(args.radius), k_str, int(args.min_radius), args.coverage_mode)

    viz_dir = args.viz_dir  # e.g., "visualizations"
    os.makedirs(viz_dir, exist_ok=True)

    # 7) Save summary JSON (under visualizations/)
    summary_path = os.path.join(viz_dir, names["summary_json"])
    summary_payload = {
        "fp_type": args.fp_type,
        "radius": int(args.radius),
        "min_radius": int(args.min_radius),
        "coverage_mode": args.coverage_mode,
        "out_dim_requested": out_dim,
        "effective_out_dim_after_min_radius": int(effective_k),
        "num_molecules": int(num_rows),
        "mean": mean_cov,
        "min": min_cov,
        "max": max_cov,
        "zero_coverage_count": zero_count,
        "first_zero_coverage_index": first_zero_retrieval_index,
        "first_zero_coverage_smiles": first_zero_smiles,
        "binary_presence_semantics": True
    }
    with open(summary_path, "w") as f:
        json.dump(summary_payload, f, indent=2)

    # 8) Emit histogram (JSON + CSV) to visualizations/
    bins = max(1, int(args.hist_bins))
    counts, bin_edges = np.histogram(cov_array, bins=bins, range=(0.0, 1.0))
    hist_json = {
        "bins": int(bins),
        "bin_edges": bin_edges.tolist(),   # length bins+1
        "counts": counts.tolist(),         # length bins
        "num_molecules": int(num_rows),
        "radius": int(args.radius),
        "min_radius": int(args.min_radius),
        "effective_K": int(effective_k),
        "fp_type": args.fp_type,
        "yscale": "log",
        "coverage_mode": args.coverage_mode,
    }
    with open(os.path.join(viz_dir, names["hist_json"]), "w") as f:
        json.dump(hist_json, f, indent=2)

    # CSV: bin_left,bin_right,count
    hist_csv_path = os.path.join(viz_dir, names["hist_csv"])
    with open(hist_csv_path, "w") as f:
        f.write("bin_left,bin_right,count\n")
        for i in range(len(counts)):
            f.write(f"{bin_edges[i]:.6f},{bin_edges[i+1]:.6f},{int(counts[i])}\n")

    # 9) Emit zero-coverage list (TXT) to visualizations/
    zero_txt_path = os.path.join(viz_dir, names["zero_txt"])
    with open(zero_txt_path, "w") as f:
        f.write(f"# Zero-coverage molecules: {zero_count} / {num_rows}\n")
        f.write(f"# Mode: {args.coverage_mode} | R<={args.radius} | minR={args.min_radius} | K={k_str}\n")
        f.write(f"# Format: [retrieval_index] SMILES\n")
        for i in zero_indices:
            f.write(f"[{ordered_keys[i]}] {ordered_smiles[i]}\n")

    # 10) Matplotlib histogram PNG (log-scaled counts)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.hist(cov_array, bins=bins, range=(0.0, 1.0))
    ax.set_yscale("log")           # log counts
    ax.set_xlabel("Atom coverage")
    ax.set_ylabel("Molecule count")
    ax.set_title(f"Coverage Histogram — {args.fp_type} (R≤{args.radius}, minR={args.min_radius}, K={k_str}, mode={args.coverage_mode})")
    fig.tight_layout()
    hist_png_path = os.path.join(viz_dir, names["hist_png"])
    fig.savefig(hist_png_path)
    plt.close(fig)

    # 11) Console summary
    print(f"[count_coverage] mode={args.coverage_mode}  mean={mean_cov:.4f}  min={min_cov:.4f}  max={max_cov:.4f}")
    print(f"[count_coverage] zero-coverage={zero_count}/{num_rows}")
    if first_zero_smiles is not None:
        print(f"[count_coverage] First zero-coverage → idx={first_zero_retrieval_index}  SMILES={first_zero_smiles}")
    print(f"[count_coverage] Saved summary:   {summary_path}")
    print(f"[count_coverage] Saved hist JSON: {os.path.join(viz_dir, names['hist_json'])}")
    print(f"[count_coverage] Saved hist CSV:  {hist_csv_path}")
    print(f"[count_coverage] Saved hist PNG:  {hist_png_path}")
    print(f"[count_coverage] Saved zero list: {zero_txt_path}")
    print(f"[count_coverage] Done in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    # If RDKit+mp gives trouble on your platform, consider:
    # mp.set_start_method("spawn", force=True)
    main()
