#!/usr/bin/env python3
"""
Per-superclass averages for:
  - # of carbons (from SMILES)
  - # of hydrogens (from SMILES with explicit Hs)
  - # of HSQC / C-NMR / H-NMR entries (from type-indicator 0/1/2)

Details:
- Each item can have multiple superclasses. This script contributes that item's
  counts to *each* of its labels (multi-label replication).
- If an item has no labels, it is assigned to the 'Unknown' bucket.
- Averages for carbons/hydrogens are computed over items with parseable SMILES
  within each superclass (reported via n_valid_smiles).
- NMR entry means are computed over all items in that superclass (n_items).

Output:
  results/analysis/per_superclass_atom_nmr_avgs.csv with columns:
    superclass,n_items,n_valid_smiles,mean_carbons,mean_hydrogens,mean_hsqc,mean_cnmr,mean_hnmr
"""

from pathlib import Path
import sys
from collections import defaultdict
from typing import Optional, Tuple, Iterable

import numpy as np
import torch

# --- Project imports (match your repo layout) ---
from archived_code.self_attention.src.settings import Args
from archived_code.self_attention.src.fp_loaders import get_fp_loader
from archived_code.self_attention.src.dataset import MoonshotDataModule

# RDKit for SMILES parsing
try:
    from rdkit import Chem
except ImportError:
    print("[ERROR] RDKit is required. Install with: conda install -c rdkit rdkit")
    sys.exit(1)

# ======================
# CONFIG
# ======================
DATA_ROOT = '/data/nas-gpu/wang/atong/MoonshotDataset'  # edit if needed
INPUT_TYPES = ['hsqc', 'h_nmr', 'c_nmr', 'mw']
OUT_CSV = Path('results/analysis/per_superclass_atom_nmr_avgs.csv')


def load_datamodule():
    args = Args(**{'data_root': DATA_ROOT, 'input_types': INPUT_TYPES})
    fp_loader = get_fp_loader(args)
    dm = MoonshotDataModule(args, str(OUT_CSV.parent), fp_loader)
    dm.setup('fit'); dm.setup('test')
    return dm


def get_superclasses(dm, idx) -> Iterable[str]:
    """
    Returns iterable of superclass labels for item idx.
    Falls back to ['Unknown'] if empty/missing.
    """
    labels = dm.test.data[idx][1].get('np_superclass', []) or []
    if not labels:
        return ['Unknown']
    return labels


def smiles_counts(smiles: str) -> Optional[Tuple[int, int]]:
    """
    Returns (nC, nH_total) for a SMILES string.
    - nC: # of carbon atoms (atomic number 6)
    - nH_total: total hydrogens after Chem.AddHs
    Returns None if parsing fails or smiles is empty.
    """
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    nC = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
    molH = Chem.AddHs(mol)
    nH = sum(1 for a in molH.GetAtoms() if a.GetAtomicNum() == 1)
    return nC, nH


def count_from_type_indicator(ti) -> Tuple[int, int, int]:
    """
    Count entries by type from a per-item type-indicator array/tuple/tensor:
      0 -> HSQC, 1 -> C-NMR, 2 -> H-NMR
    Returns (n_hsqc, n_c, n_h).
    """
    if ti is None:
        return 0, 0, 0
    try:
        if torch.is_tensor(ti):
            ti_flat = ti.view(-1) if ti.ndim else ti.view(1)
            n_hsqc = int((ti_flat == 0).sum().item())
            n_c    = int((ti_flat == 1).sum().item())
            n_h    = int((ti_flat == 2).sum().item())
            return n_hsqc, n_c, n_h
        arr = np.asarray(ti).reshape(-1)
        return int((arr == 0).sum()), int((arr == 1).sum()), int((arr == 2).sum())
    except Exception:
        return 0, 0, 0


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    dm = load_datamodule()

    N = len(dm.test)
    print(dm.test[0][1])
    if N == 0:
        print("[WARN] Test set is empty; nothing to compute.")
        return

    # Per-class aggregators
    sums = defaultdict(lambda: {
        'n_items': 0,
        'n_valid_smiles': 0,
        'sum_C': 0.0,
        'sum_H': 0.0,
        'sum_hsqc': 0.0,
        'sum_cnmr': 0.0,
        'sum_hnmr': 0.0,
    })

    for idx in range(N):
        # Type-indicator counts
        try:
            ti = dm.test[idx][2]
        except Exception:
            ti = None
        n_hsqc_i, n_c_i, n_h_i = count_from_type_indicator(ti)

        # Atom counts from SMILES
        smiles = dm.test.data[idx][1].get('smiles', None)
        atom_counts = smiles_counts(smiles)
        has_valid_smiles = atom_counts is not None
        if has_valid_smiles:
            nC, nH = atom_counts

        # Superclass labels
        labels = list(get_superclasses(dm, idx))

        # Update each superclass bucket
        for cls in labels:
            agg = sums[cls]
            agg['n_items'] += 1
            agg['sum_hsqc'] += n_hsqc_i
            agg['sum_cnmr'] += n_c_i
            agg['sum_hnmr'] += n_h_i
            if has_valid_smiles:
                agg['n_valid_smiles'] += 1
                agg['sum_C'] += nC
                agg['sum_H'] += nH

        # Progress ping
        if (idx + 1) % 1000 == 0 or (idx + 1) == N:
            print(f"[INFO] Processed {idx + 1}/{N}")

    # Write per-class CSV
    # Sort classes by n_items desc
    rows = []
    for cls, agg in sorted(sums.items(), key=lambda kv: kv[1]['n_items'], reverse=True):
        n_items = agg['n_items']
        n_valid = agg['n_valid_smiles']
        mean_carbons   = (agg['sum_C'] / n_valid) if n_valid > 0 else float('nan')
        mean_hydrogens = (agg['sum_H'] / n_valid) if n_valid > 0 else float('nan')
        mean_hsqc = agg['sum_hsqc'] / n_items if n_items > 0 else float('nan')
        mean_cnmr = agg['sum_cnmr'] / n_items if n_items > 0 else float('nan')
        mean_hnmr = agg['sum_hnmr'] / n_items if n_items > 0 else float('nan')

        rows.append((
            cls,
            n_items,
            n_valid,
            mean_carbons,
            mean_hydrogens,
            mean_hsqc,
            mean_cnmr,
            mean_hnmr,
        ))

    with open(OUT_CSV, 'w') as f:
        f.write("superclass,n_items,n_valid_smiles,mean_carbons,mean_hydrogens,mean_hsqc,mean_cnmr,mean_hnmr\n")
        for r in rows:
            cls, n_items, n_valid, mc, mh, hsqc, cnmr, hnmr = r
            f.write(f"{cls},{n_items},{n_valid},"
                    f"{(f'{mc:.6f}' if np.isfinite(mc) else 'nan')},"
                    f"{(f'{mh:.6f}' if np.isfinite(mh) else 'nan')},"
                    f"{hsqc:.6f},{cnmr:.6f},{hnmr:.6f}\n")

    print(f"[INFO] Wrote CSV → {OUT_CSV.resolve()}")
    print(f"[INFO] Classes written: {len(rows)}")


if __name__ == "__main__":
    main()
