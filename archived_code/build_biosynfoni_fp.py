#!/usr/bin/env python3
"""
Build 39-D Biosynfoni fingerprints (log1p counts) for training + retrieval.

Outputs under DATASET_ROOT/Biosynfoni:
  - rankingset.pt           (N_retrieval, 39) float32, log1p-transformed
  - smiles_to_row.json      SMILES -> row index in rankingset.pt
  - train_fps.pt            (N_train, 39) float32, log1p-transformed
  - train_indices.pt        (N_train,) long tensor of idxs aligned with train_fps.pt

Inputs (pickles of dict[int, dict]):
  - DATASET_ROOT/index.pkl                 -> training/val/test pool (has ["smiles"], ["split"])
  - DATASET_ROOT/rankingset_meta.pkl       -> retrieval pool (has ["smiles"])

Notes:
  - We use *dense* tensors. 39 x ~500k ~= 78 MB (float32) — fine in memory.
  - No row normalization here; Tanimoto will use raw log-count vectors.
"""

import argparse, json, os, pickle
import torch
from rdkit import Chem
from tqdm import tqdm

# You already have this installed/available
from biosynfoni import Biosynfoni

def smiles_to_vec(smi: str) -> torch.Tensor:
    """Return (39,) float32 vector as log1p(counts). If failure, returns zeros."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return torch.zeros(39, dtype=torch.float32)
        fp_list = Biosynfoni(mol).fingerprint  # python list length 39
        # Convert to log1p(counts)
        v = torch.tensor(fp_list, dtype=torch.float32)
        v = torch.log1p(v)
        return v
    except Exception:
        return torch.zeros(39, dtype=torch.float32)

def build_rankingset(dataset_root: str, out_dir: str) -> None:
    meta_path = os.path.join(dataset_root, "rankingset_meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    # meta: dict[idx] -> {"smiles": ..., ...}
    # fix a stable row order
    idxs = sorted(meta.keys())
    rows = []
    smiles_to_row = {}
    for row, idx in tqdm(enumerate(idxs)):
        smi = meta[idx]["smiles"]
        rows.append(smiles_to_vec(smi))
        smiles_to_row[smi] = row

    X = torch.stack(rows, dim=0) if rows else torch.zeros(0, 39, dtype=torch.float32)

    os.makedirs(out_dir, exist_ok=True)
    torch.save(X, os.path.join(out_dir, "rankingset.pt"))
    with open(os.path.join(out_dir, "smiles_to_row.json"), "w") as f:
        json.dump(smiles_to_row, f)

    print(f"[Biosynfoni] rankingset.pt saved: {tuple(X.shape)} at {out_dir}")

def build_train_set(dataset_root: str, out_dir: str) -> None:
    idx_path = os.path.join(dataset_root, "index.pkl")
    with open(idx_path, "rb") as f:
        data = pickle.load(f)

    # Keep *all* indices found in index.pkl (or filter to split == 'train' if you prefer)
    idxs = sorted(data.keys())

    rows = []
    kept = []
    for idx in tqdm(idxs):
        smi = data[idx]["smiles"]
        v = smiles_to_vec(smi)
        rows.append(v)
        kept.append(idx)

    Xtr = torch.stack(rows, dim=0) if rows else torch.zeros(0, 39, dtype=torch.float32)
    Itr = torch.tensor(kept, dtype=torch.long)

    os.makedirs(out_dir, exist_ok=True)
    torch.save(Xtr, os.path.join(out_dir, "train_fps.pt"))
    torch.save(Itr, os.path.join(out_dir, "train_indices.pt"))

    print(f"[Biosynfoni] train_fps.pt saved: {tuple(Xtr.shape)}; aligned indices: {len(Itr)}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--out_subdir", default="Biosynfoni",
                   help="Folder inside DATASET_ROOT for outputs")
    args = p.parse_args()

    out_dir = os.path.join(args.dataset_root, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    build_rankingset(args.dataset_root, out_dir)
    build_train_set(args.dataset_root, out_dir)

if __name__ == "__main__":
    main()
