import os, json, argparse, pickle, math
from typing import Dict, Tuple, List, Iterable
import numpy as np
import torch
from rdkit import Chem
from tqdm import tqdm

Feature = Tuple[int, str, str, int]  # (bit_id, atom_symbol, frag_smiles, radius)

def load_meta(root):
    with open(os.path.join(root, "meta.pkl"), "rb") as f:
        return pickle.load(f)

def load_vocab(root):
    with open(os.path.join(root, "vocab.pkl"), "rb") as f:
        return pickle.load(f)

def ensure_rdkit(radius):
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius)
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    return gen, ao

def largest_fragment(m):
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize as rs
    except Exception:
        try:
            import rdkit.Chem.MolStandardize as rs
        except Exception:
            rs = None
    if rs is None: return m
    try:
        return rs.LargestFragmentChooser().choose(m)
    except Exception:
        return m

def frag_counts(mol, radii, gen, ao) -> Dict[Feature, int]:
    _ = gen.GetSparseFingerprint(mol, additionalOutput=ao)
    info = ao.GetBitInfoMap()
    feats = {}
    for bit_id, envs in info.items():
        for atom_idx, r in envs:
            if r not in radii: continue
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, r, atom_idx)
            sub = Chem.PathToSubmol(mol, env)
            frag = Chem.MolToSmiles(sub, canonical=True)
            atom = mol.GetAtomWithIdx(atom_idx).GetSymbol()
            key = (int(bit_id), atom, frag, int(r))
            feats[key] = feats.get(key, 0) + 1
    return feats

def vectorize(feats: Dict[Feature,int], meta, vocab, normalize_blocks=True) -> np.ndarray:
    D = int(meta["D_total"])
    v = np.zeros(D, dtype=np.float32)

    # global
    lo, hi = meta["block_offsets"]["global"]
    g = np.zeros(hi-lo, dtype=np.float32)
    idx_g = vocab["index_global"]; idf_g = vocab["idf_global"]
    tf_mode = meta["settings"]["tf_mode"]
    for f, c in feats.items():
        j = idx_g.get(f)
        if j is not None:
            g[j] = (1.0 if tf_mode=="binary" else math.log1p(c)) * float(idf_g[f])
    if normalize_blocks and np.linalg.norm(g) > 0:
        g /= max(1e-12, np.linalg.norm(g))
    v[lo:hi] = g

    # classes
    for key,(lo,hi) in meta["block_offsets"].items():
        if not key.startswith("class::"): continue
        cls = key.split("class::",1)[1]
        idx_c = vocab["index_class"].get(cls, {})
        idf_c = vocab["idf_class"].get(cls, {})
        if not idx_c: continue
        b = np.zeros(hi-lo, dtype=np.float32)
        for f, c in feats.items():
            j = idx_c.get(f)
            if j is not None:
                b[j] = (1.0 if tf_mode=="binary" else math.log1p(c)) * float(idf_c[f])
        if normalize_blocks and np.linalg.norm(b) > 0:
            b /= max(1e-12, np.linalg.norm(b))
        v[lo:hi] = np.maximum(v[lo:hi], b)
    return v

def write_bank(fp_root_split: str, idxs_in_order: List[int], rows: List[torch.Tensor], index_map: Dict[int, dict]):
    os.makedirs(fp_root_split, exist_ok=True)
    X = torch.stack(rows, dim=0)  # (N, D)
    torch.save(X, os.path.join(fp_root_split, "fingerprints.pt"))
    Xn = X / torch.linalg.norm(X, dim=1, keepdim=True).clamp_min(1e-12)
    torch.save(Xn, os.path.join(fp_root_split, "rankingset.pt"))
    # smiles maps in bank order
    smi2idx = {index_map[i]["smiles"]: k for k, i in enumerate(idxs_in_order)}
    idx2smi = {k: index_map[i]["smiles"] for k, i in enumerate(idxs_in_order)}
    with open(os.path.join(fp_root_split, "smiles_to_idx.json"), "w") as f: json.dump(smi2idx, f)
    with open(os.path.join(fp_root_split, "idx_to_smiles.json"), "w") as f: json.dump(idx2smi, f)

def main(dataset_root: str, fp_type: str):
    fp_root = os.path.join(dataset_root, fp_type)
    meta  = load_meta(fp_root)
    vocab = load_vocab(fp_root)
    radii = tuple(meta["settings"]["radii"])
    gen, ao = ensure_rdkit(max(radii))

    # Load global index with splits
    with open(os.path.join(dataset_root, "index.pkl"), "rb") as f:
        index = pickle.load(f)  # {idx: {"smiles":..., "split": "train"/"val"/"test", ...}}

    # Group idxs by split (train/val/test); also keep an "all" group
    groups = {"train": [], "val": [], "test": [], "all": []}
    for k, rec in index.items():
        i = int(k)
        split = str(rec.get("split", "train"))
        if split not in groups:  # guard unknown labels
            split = "train"
        groups[split].append(i)
        groups["all"].append(i)

    # For each split: save per-idx fp/{idx}.pt and split-local banks
    for split, idxs in groups.items():
        idxs = sorted(idxs)
        fp_dir = os.path.join(fp_root, split, "fp")
        os.makedirs(fp_dir, exist_ok=True)

        rows, order = [], []
        for i in tqdm(idxs):
            smi = index[i]["smiles"]
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mol = largest_fragment(mol)
            feats = frag_counts(mol, radii, gen, ao)
            v = vectorize(feats, meta, vocab, normalize_blocks=meta["settings"]["normalize_blocks"])
            t = torch.tensor(v, dtype=torch.float32)
            # write per-idx
            torch.save(t, os.path.join(fp_dir, f"{i}.pt"))
            rows.append(t); order.append(i)

        # write split banks (fingerprints.pt, rankingset.pt, smiles maps)
        if rows:
            write_bank(os.path.join(fp_root, split), order, rows, index)

        print(f"[{fp_type}] wrote {len(order)} vectors for split='{split}' under {os.path.join(fp_root, split)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Materialize IRFP per-idx files and per-split banks")
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--fp_type", required=True, help="RankingBalanced / RankingGlobal / RankingSuperclass")
    args = ap.parse_args()
    main(args.dataset_root, args.fp_type)
