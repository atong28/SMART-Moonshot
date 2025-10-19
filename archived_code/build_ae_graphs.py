# ae_build.py
import os
import pickle
import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

# ---------- MoleMCL-compatible featurization ----------

allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)),
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ],
}

def mol_to_graph_data_obj_simple(mol: Chem.Mol) -> Data:
    # Atoms
    atom_feats = []
    for a in mol.GetAtoms():
        atom_feats.append([
            allowable_features['possible_atomic_num_list'].index(a.GetAtomicNum()),
            allowable_features['possible_chirality_list'].index(a.GetChiralTag()),
        ])
    x = torch.tensor(np.array(atom_feats), dtype=torch.long)

    # Bonds
    if mol.GetNumBonds() > 0:
        edges, eattrs = [], []
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            feat = [
                allowable_features['possible_bonds'].index(b.GetBondType()),
                allowable_features['possible_bond_dirs'].index(b.GetBondDir()),
            ]
            edges.append((i, j)); eattrs.append(feat)
            edges.append((j, i)); eattrs.append(feat)
        edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
        edge_attr  = torch.tensor(np.array(eattrs), dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, 2), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# ---------- Canonicalization helpers (drop stereo, drop multi-fragment) ----------

def canonicalize_smiles_no_stereo(smiles: str) -> Optional[str]:
    """Return canonical SMILES **without stereochemistry**; None if invalid."""
    if '.' in smiles:  # drop multi-fragment molecules
        return None
    mol = AllChem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.MolToSmiles(mol, isomericSmiles=False)

def smiles_to_mol_no_stereo(smiles: str) -> Optional[Chem.Mol]:
    """Parse → de-stereo canonicalize → reparse for consistency."""
    smi_ns = canonicalize_smiles_no_stereo(smiles)
    if smi_ns is None:
        return None
    return AllChem.MolFromSmiles(smi_ns)

# ---------- Split utilities ----------

def load_spectre_split_map(spectre_index_path: str) -> Dict[str, str]:
    """
    Load {root}/index.pkl from the SPECTRE+DiffMS dataset and build
    a map: smiles_no_stereo -> split ('train'/'val'/'test').

    Expects each entry to have a 'smiles' and 'split' field.
    """
    with open(spectre_index_path, "rb") as f:
        spectre_idx: Dict[int, dict] = pickle.load(f)

    smi2split: Dict[str, str] = {}
    for _, entry in spectre_idx.items():
        smi_raw = entry.get("smiles")
        split = entry.get("split")
        if not smi_raw or not split:
            continue
        smi_ns = canonicalize_smiles_no_stereo(smi_raw)
        if smi_ns is None:
            continue
        smi2split[smi_ns] = split
    return smi2split

def assign_random_split(smi: str, rng: random.Random, p=(0.8, 0.1, 0.1)) -> str:
    """Deterministic split assignment using a seeded RNG (by caller)."""
    r = rng.random()
    if r < p[0]:
        return "train"
    elif r < p[0] + p[1]:
        return "val"
    else:
        return "test"

# ---------- Builder ----------

def build_ae_graphs_from_metadata(
    metadata_pkl: str,
    out_root: str,
    spectre_root: str,
    smiles_index_in_tuple: int = 0,
    start_idx: int = 0,
    limit: Optional[int] = None,
    seed: int = 42,
) -> Dict[int, dict]:
    """
    Build AE graphs and split index.

    Inputs:
      - metadata_pkl: list of tuples like (smiles, name, mw, source, ...)
      - out_root: your dataset root (we will create {out_root}/AE/*)
      - spectre_root: path where SPECTRE+DiffMS has its index.pkl (at {spectre_root}/index.pkl)
      - smiles_index_in_tuple: which element in tuple is the SMILES
      - start_idx: starting index for AE index
      - limit: cap how many metadata rows to process
      - seed: RNG seed for deterministic random splits

    Outputs:
      - Writes {out_root}/AE/index.pkl as: { idx: {'smiles': smi_no_stereo, 'split': split} }
      - Writes graphs to {out_root}/AE/Graphs/{idx}.pt
      - Returns the index mapping (same object written to index.pkl)
    """
    with open(metadata_pkl, "rb") as f:
        meta_list = pickle.load(f)
    if limit is not None:
        meta_list = meta_list[:limit]

    spectre_index_path = os.path.join(spectre_root, "index.pkl")
    smi2split_spectre = load_spectre_split_map(spectre_index_path)

    ae_dir = os.path.join(out_root, "AE")
    graphs_dir = os.path.join(ae_dir, "Graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    # Deterministic RNG for splits: seed + smiles hash keeps it stable
    base_rng = random.Random(seed)

    index_map: Dict[int, dict] = {}
    num_ok = num_fail = 0
    split_counts = {"train": 0, "val": 0, "test": 0, "test_final": 0}

    for rel_i, tup in tqdm(enumerate(meta_list)):
        idx = start_idx + rel_i
        raw_smiles = tup[smiles_index_in_tuple]

        # 1) Canonicalize & drop stereo; drop multi-fragment
        smi_ns = canonicalize_smiles_no_stereo(raw_smiles)
        if smi_ns is None:
            num_fail += 1
            continue

        # 2) Choose split: mirror SPECTRE if present, else deterministic random
        split = smi2split_spectre.get(smi_ns)
        if split is None:
            # derive a per-smiles RNG so assignment is stable across runs
            local_seed = hash((smi_ns, seed)) & 0xFFFFFFFF
            rng = random.Random(local_seed)
            split = assign_random_split(smi_ns, rng)

        # 3) Build PyG graph
        mol = AllChem.MolFromSmiles(smi_ns)
        if mol is None or mol.GetNumAtoms() == 0:
            num_fail += 1
            continue

        try:
            data = mol_to_graph_data_obj_simple(mol)
            torch.save(data, os.path.join(graphs_dir, f"{idx}.pt"))
            index_map[idx] = {"smiles": smi_ns, "split": split}
            split_counts[split] += 1
            num_ok += 1
        except Exception:
            num_fail += 1
            continue

    # 4) Write AE/index.pkl
    with open(os.path.join(ae_dir, "index.pkl"), "wb") as f:
        pickle.dump(index_map, f)

    print(f"[AE] Wrote {num_ok} graphs, {num_fail} failures → {ae_dir}")
    print(f"[AE] Split counts: {split_counts}")
    return index_map


index_map = build_ae_graphs_from_metadata(
    metadata_pkl="/data/nas-gpu/wang/atong/MoonshotDataset/rankingset_metadata.pkl",
    out_root="/data/nas-gpu/wang/atong/MoonshotDataset",           # will create dataset/AE/...
    spectre_root="/data/nas-gpu/wang/atong/MoonshotDataset",       # expects dataset/index.pkl from SPECTRE
    smiles_index_in_tuple=0,
    seed=0,
)
