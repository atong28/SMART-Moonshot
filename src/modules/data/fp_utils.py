# fp_utils.py
from __future__ import annotations

import os
import json
import math
import pickle
import multiprocessing as mp
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple, Any, Union

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from tqdm import tqdm

# ---------------------------
# Types
# ---------------------------
BitInfo = Tuple[int, str, str, int]  # (bit_id, atom_symbol, fragment_smiles, radius)

G_RADIUS = None
G_MAPPING = None  # for CSR bitinfo_to_col

def _init_count(radius: int):
    global G_RADIUS
    G_RADIUS = radius

def _worker_count_one(smi: str):
    # uses G_RADIUS set by _init_count
    return count_circular_substructures(smi, G_RADIUS)

def _init_csr(radius: int, mapping: Dict[BitInfo, int]):
    global G_RADIUS, G_MAPPING
    G_RADIUS = radius
    G_MAPPING = mapping

def _worker_row_nonzeros(args):
    # args: (row_idx, smi)
    row_idx, smi = args
    present = count_circular_substructures(smi, G_RADIUS)
    cols = []
    get = G_MAPPING.get
    for b in present.keys():
        col = get(b)
        if col is not None:
            cols.append(col)
    cols = sorted(set(cols))
    return row_idx, cols

# ---------------------------
# IO helpers
# ---------------------------
def _load_as_dict(path: str) -> Any:
    with open(path, "rb") as f:
        if path.endswith(".pkl") or path.endswith(".pickle"):
            return pickle.load(f)
    with open(path, "r") as f:
        return json.load(f)


def load_smiles_index(path: str) -> Dict[int, str]:
    """
    Load an idx->smiles mapping from .json or .pkl.

    Accepts one of:
      - List[Dict[...]] with a smiles-like field
      - Dict[int, Dict[...]] with a smiles-like field
      - Dict[str|int, str] direct mapping to smiles

    Recognized smiles fields (first found wins): 'smiles', 'canonical_2d_smiles'
    """
    data = _load_as_dict(path)

    def _extract_smiles(rec: Any) -> Optional[str]:
        if isinstance(rec, str):
            return rec
        if isinstance(rec, dict):
            for key in ("smiles", "canonical_2d_smiles"):
                v = rec.get(key)
                if isinstance(v, str) and v:
                    return v
        return None

    smiles_map: Dict[int, str] = {}
    if isinstance(data, list):
        for idx, rec in enumerate(data):
            s = _extract_smiles(rec)
            if s:
                smiles_map[idx] = s
    elif isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, str):
                s = v
            else:
                s = _extract_smiles(v)
            if s:
                try:
                    idx = int(k)
                except Exception:
                    continue
                smiles_map[idx] = s
    else:
        raise ValueError(f"Unsupported index structure in {path}")
    if not smiles_map:
        raise ValueError(f"No smiles found in {path}")
    return smiles_map


# ---------------------------
# Morgan / fragments
# ---------------------------
def _mk_rdkit(radius: int):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius)
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    return gen, ao


def get_bitinfos(
    smiles: str, radius: int, ignore_atoms: Optional[Iterable[int]] = None
) -> Tuple[Optional[Dict[int, List[BitInfo]]], Optional[Set[BitInfo]]]:
    """
    Extract Morgan bit environments for each atom index (for a specific radius).
    """
    ignore_atoms = tuple(ignore_atoms or ())
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    gen, ao = _mk_rdkit(radius)
    _ = gen.GetSparseFingerprint(mol, additionalOutput=ao)
    info = ao.GetBitInfoMap()

    atom_to_bit_infos: Dict[int, List[BitInfo]] = defaultdict(list)
    all_bit_infos: Set[BitInfo] = set()

    for bit_id, atom_envs in info.items():
        for atom_idx, curr_radius in atom_envs:
            if atom_idx in ignore_atoms:
                continue
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, curr_radius, atom_idx)
            submol = Chem.PathToSubmol(mol, env)
            frag_smiles = Chem.MolToSmiles(submol, canonical=True)
            atom_sym = mol.GetAtomWithIdx(atom_idx).GetSymbol()
            bit_info: BitInfo = (bit_id, atom_sym, frag_smiles, curr_radius)
            atom_to_bit_infos[atom_idx].append(bit_info)
            all_bit_infos.add(bit_info)

    return atom_to_bit_infos, all_bit_infos


def count_circular_substructures(
    smiles: str, radius: int, ignore_atoms: Optional[Iterable[int]] = None
) -> Dict[BitInfo, int]:
    """
    Return presence map BitInfo -> 1 for a SMILES at given radius.
    """
    bit_info_counter: Dict[BitInfo, int] = defaultdict(int)
    atom_to_bit_infos, all_bit_infos = get_bitinfos(smiles, radius, ignore_atoms or ())
    if atom_to_bit_infos is None:
        return bit_info_counter
    for bit_info in all_bit_infos:
        bit_info_counter[bit_info] = 1
    return bit_info_counter


def merge_counts(counts_list: Iterable[Dict[BitInfo, int]]) -> Counter:
    total_count = Counter()
    for count in counts_list:
        total_count.update(count)
    return total_count


# ---------------------------
# Entropy
# ---------------------------
def compute_entropy(counts: np.ndarray, total_dataset_size: int) -> np.ndarray:
    """
    Standard binary entropy (positive):
      H(p) = - [ p log2 p + (1-p) log2 (1-p) ]
    """
    p = np.clip(counts / float(total_dataset_size), 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


# ---------------------------
# Training fragments & retrieval counting
# ---------------------------
def _save_fragments_for_idx(args) -> None:
    idx, smiles, out_dir, radius = args
    frags = list(count_circular_substructures(smiles, radius).keys())
    torch.save(frags, os.path.join(out_dir, f"{idx}.pt"))


def generate_fragments_for_training(
    index_path: str,
    out_dir: str,
    radius: int,
    num_procs: int = 0,
) -> None:
    """
    Build per-idx fragment lists under out_dir/Fragments/{idx}.pt using index_path (training set).
    """
    smiles_map = load_smiles_index(index_path)
    frag_dir = os.path.join(out_dir, "Fragments")
    os.makedirs(frag_dir, exist_ok=True)

    items = [(idx, smi, frag_dir, radius) for idx, smi in smiles_map.items()]
    procs = (mp.cpu_count() if not num_procs else max(1, int(num_procs)))
    with mp.Pool(processes=procs) as pool:
        for _ in tqdm(
            pool.imap_unordered(_save_fragments_for_idx, items),
            total=len(items),
            desc="Saving training fragments",
        ):
            pass


def count_fragments_over_retrieval(
    retrieval_path: str,
    radius: int,
    num_procs: int = 0,
) -> Counter:
    """
    Return Counter[BitInfo] over the retrieval set (presence in #molecules).
    """
    smiles_map = load_smiles_index(retrieval_path)
    smiles_list = list(smiles_map.values())
    procs = (mp.cpu_count() if not num_procs else max(1, int(num_procs)))

    counts: List[Dict[BitInfo, int]] = []
    if procs == 1:
        # serial fallback (useful for debugging)
        for smi in tqdm(smiles_list, total=len(smiles_list), desc="Counting retrieval fragments"):
            counts.append(count_circular_substructures(smi, radius))
    else:
        with mp.Pool(processes=procs, initializer=_init_count, initargs=(radius,)) as pool:
            for c in tqdm(
                pool.imap_unordered(_worker_count_one, smiles_list, chunksize=64),
                total=len(smiles_list),
                desc="Counting retrieval fragments",
            ):
                counts.append(c)
    return merge_counts(counts)


def write_counts(counter: Counter, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(counter, f)


# ---------------------------
# Rankingset (CSR)
# ---------------------------
def build_rankingset_csr(
    retrieval_path: str,
    bitinfo_to_col: Dict[BitInfo, int],
    radius: int,
    num_procs: int = 0,
) -> torch.Tensor:
    """
    Build a torch.sparse_csr_tensor with shape (num_retrieval, num_features)
    where values are 0/1 presence for each retrieval SMILES.

    Rows are L2-normalized to unit length (so cosine similarity = dot product).
    """
    smiles_map = load_smiles_index(retrieval_path)
    ordered = sorted(smiles_map.items(), key=lambda kv: kv[0])  # consistent row order
    num_rows = len(ordered)
    num_cols = max(bitinfo_to_col.values()) + 1 if bitinfo_to_col else 0

    procs = (mp.cpu_count() if not num_procs else max(1, int(num_procs)))
    rows_cols: List[Tuple[int, List[int]]] = []

    if procs == 1:
        for res in tqdm(
            ((i, smi) for i, (_, smi) in enumerate(ordered)),
            total=num_rows,
            desc="Building CSR rows",
        ):
            rows_cols.append(_worker_row_nonzeros(res))
    else:
        with mp.Pool(
            processes=procs,
            initializer=_init_csr,
            initargs=(radius, bitinfo_to_col),
        ) as pool:
            for res in tqdm(
                pool.imap_unordered(
                    _worker_row_nonzeros,
                    [(i, smi) for i, (_, smi) in enumerate(ordered)],
                    chunksize=64,
                ),
                total=num_rows,
                desc="Building CSR rows",
            ):
                rows_cols.append(res)

    # Reassemble in row order
    rows_cols.sort(key=lambda x: x[0])

    crow_indices = [0]
    col_indices: List[int] = []
    values: List[float] = []
    nnz_so_far = 0

    for _, cols in rows_cols:
        if not cols:
            crow_indices.append(nnz_so_far)
            continue
        nnz = len(cols)
        col_indices.extend(cols)
        inv_len = 1.0 / math.sqrt(nnz)
        values.extend([inv_len] * nnz)
        nnz_so_far += nnz
        crow_indices.append(nnz_so_far)

    crow = torch.tensor(crow_indices, dtype=torch.int64)
    cols = torch.tensor(col_indices, dtype=torch.int64)
    vals = torch.tensor(values, dtype=torch.float32)
    return torch.sparse_csr_tensor(crow, cols, vals, size=(num_rows, num_cols))


