"""
Information-Rich Fingerprint (IRFP) Builder — Exclusive Class Bits
------------------------------------------------------------------

Builds a hybrid fingerprint with:
  • Global TF–IDF block (shared across all molecules)
  • Per-superclass class-enriched TF–IDF blocks (diagnostic features)

Key upgrade: **exclusive class assignment** — each fragment feature is assigned
to at most ONE superclass (the one where its log-enrichment is highest), so no
feature is duplicated across class blocks. Optionally keep class-selected
features OUT of the global block to avoid double-counting.

Supports:
  • Binary vs log-count TF
  • Multi-radius (e.g., 1,2,3)
  • Class vocab sizing: fixed per-class OR distributed from a total budget via
    sqrt(class size) with min/max caps
  • Missing / multi-label classes
  • Largest-organic-fragment cleanup (drops salts/solvents)
  • Per-block L2 normalization; optional α/β weighting
  • Optionally emit a ready-to-use **rankingset.pt** (row-normalized) for your ranker

Input JSON formats:
  • Dict:  {"SMI": "Class"} OR {"SMI": ["ClassA", "ClassB"]}
  • List:  ["SMI1", "SMI2", ...]    (treated as unlabeled; class blocks masked)

Outputs (out_dir):
  • fingerprints.pt          float32 tensor [N, D_total] (by default: per-block L2, un-weighted)
  • smiles_to_idx.json       SMILES → row index
  • idx_to_smiles.json       row index → SMILES
  • meta.pkl                 settings, classes, block offsets, sizes
  • vocab.pkl                vocabularies, IDFs, and index maps
  • (optional) rankingset.pt row-L2-normalized copy for RankingSet (if --emit_rankingset)

Dependencies: RDKit, NumPy, Torch, TQDM
"""
from __future__ import annotations
import json
import math
import os
import pickle
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Tuple, Optional, Union

import numpy as np
import torch
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

# Robust import for standardization (version-dependent in RDKit)
try:
    # RDKit 2020+ typical path
    from rdkit.Chem.MolStandardize import rdMolStandardize
except Exception:  # pragma: no cover
    try:
        import rdkit.Chem.MolStandardize as rdMolStandardize  # older path
    except Exception:
        rdMolStandardize = None

# ---------------------------
# Feature identity (explicit)
# ---------------------------
Feature = Tuple[int, str, str, int]  # (bit_id, atom_symbol, frag_smiles, radius)


@dataclass
class Settings:
    # Fragmenting / features
    radii: Tuple[int, ...] = (1, 2, 3)
    tf_mode: str = "binary"          # 'binary' or 'log'

    # Global vocabulary
    global_vocab_size: int = 10000
    global_selection: str = "entropy" # 'entropy' or 'df_window'
    df_min_frac: float = 0.01          # for df_window
    df_max_frac: float = 0.60
    exclude_class_feats_from_global: bool = True  # avoid duplicates between blocks

    # Class vocab sizing
    class_vocab_mode: str = "sqrt_budget"  # 'fixed' or 'sqrt_budget'
    class_vocab_size: int = 256             # used if mode == 'fixed'
    class_vocab_total: int = 6000           # total budget if mode == 'sqrt_budget'
    class_k_min: int = 64
    class_k_max: int = 512

    # Enrichment / statistics
    eps: float = 0.5           # smoothing for enrichment + IDF
    min_support: int = 5       # minimum DF within class to consider feature
    enrich_tau: float = math.log(1.5)  # require at least ~1.5x odds

    # Cleanup / normalization
    keep_largest_organic_fragment: bool = True
    normalize_blocks: bool = True
    alpha: float = 1.0         # optional stored weighting for global block
    beta: float = 1.0          # optional stored weighting for sum of class blocks
    apply_weights: bool = False

    # Misc
    seed: int = 0

    # Optional: emit rankingset.pt (row-normalized)
    emit_rankingset: bool = False


class IRFPExclusiveBuilder:
    def __init__(self, smiles_and_labels: List[Tuple[str, List[str]]], settings: Optional[Settings] = None):
        self.data = smiles_and_labels  # list of (SMILES, [labels]) where labels may be []
        self.settings = settings or Settings()
        self.classes = sorted(list({lab for _, labs in self.data for lab in (labs or [])}))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        max_r = max(self.settings.radii) if self.settings.radii else 2
        self._gen = rdFingerprintGenerator.GetMorganGenerator(radius=max_r)
        self._ao = rdFingerprintGenerator.AdditionalOutput()
        self._ao.AllocateBitInfoMap()

        # Filled during fit
        self.vocab_global: List[Feature] = []
        self.idf_global: Dict[Feature, float] = {}
        self.index_global: Dict[Feature, int] = {}

        self.vocab_class: Dict[str, List[Feature]] = {}
        self.idf_class: Dict[str, Dict[Feature, float]] = {}
        self.index_class: Dict[str, Dict[Feature, int]] = {}

        self.block_offsets: Dict[str, Tuple[int, int]] = {}

    # --------------------
    # Public
    # --------------------
    def fit_and_build(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)

        # 1) Preprocess & enumerate counts
        counts_per_mol: List[Dict[Feature, int]] = []
        smiles_clean: List[str] = []
        labels_list: List[List[str]] = []
        for smi, labs in tqdm(self.data, desc="Enumerate fragments"):
            mol = self._smiles_to_mol(smi)
            if mol is None:
                continue
            if self.settings.keep_largest_organic_fragment and rdMolStandardize is not None:
                mol = self._largest_organic_fragment(mol)
                if mol is None:
                    continue
            feats = self._fragment_counts(mol)
            counts_per_mol.append(feats)
            smiles_clean.append(Chem.MolToSmiles(mol, canonical=True))
            labels_list.append(list(labs or []))

        N = len(counts_per_mol)
        assert N > 0, "No valid molecules after preprocessing"

        # 2) Document frequencies (global + per-class)
        df_global: Dict[Feature, int] = {}
        df_class: Dict[str, Dict[Feature, int]] = {c: {} for c in self.classes}
        class_sizes: Dict[str, int] = {c: 0 for c in self.classes}

        for feats, labs in tqdm(zip(counts_per_mol, labels_list), total=N, desc="Corpus stats"):
            for f in feats.keys():
                df_global[f] = df_global.get(f, 0) + 1
            for c in labs:
                class_sizes[c] += 1
                d = df_class[c]
                for f in feats.keys():
                    d[f] = d.get(f, 0) + 1

        # 3) Build exclusive class vocabs (feature → at most one class)
        self._build_class_vocabs_exclusive(df_global, df_class, N, class_sizes)

        # 4) Build global vocab (optionally exclude any class-selected features)
        self._build_global_vocab(df_global, N)

        # 5) Compute block offsets
        D_total = self._compute_block_offsets()

        # 6) Build hard-gated vectors (union for multi-label)
        fps = np.zeros((N, D_total), dtype=np.float32)
        for i, (feats, labs) in enumerate(tqdm(zip(counts_per_mol, labels_list), total=N, desc="Build FPs")):
            # Global
            vg = self._vector_global(feats)
            if self.settings.normalize_blocks:
                vg = self._l2norm(vg)
            if self.settings.apply_weights:
                vg *= self.settings.alpha
            lo_g, hi_g = self.block_offsets["global"]
            fps[i, lo_g:hi_g] = vg

            # Class union (if labeled); unlabeled -> all-zero class blocks
            for c in labs:
                if c not in self.vocab_class:
                    continue
                vc = self._vector_class_block(feats, c)
                if self.settings.normalize_blocks:
                    vc = self._l2norm(vc)
                if self.settings.apply_weights:
                    vc *= (self.settings.beta / max(1, len(labs)))  # average if multi-label
                lo_c, hi_c = self.block_offsets[f"class::{c}"]
                # Union: max to avoid double-add if multi-label shares indices (shouldn't with exclusivity, but safe)
                fps[i, lo_c:hi_c] = np.maximum(fps[i, lo_c:hi_c], vc)

        # 7) Save basic artifacts
        smiles_to_idx = {smi: i for i, smi in enumerate(smiles_clean)}
        idx_to_smiles = {i: smi for smi, i in smiles_to_idx.items()}

        torch.save(torch.from_numpy(fps), os.path.join(out_dir, "fingerprints.pt"))
        with open(os.path.join(out_dir, "smiles_to_idx.json"), "w") as f:
            json.dump(smiles_to_idx, f)
        with open(os.path.join(out_dir, "idx_to_smiles.json"), "w") as f:
            json.dump(idx_to_smiles, f)

        meta = {
            "settings": asdict(self.settings),
            "classes": self.classes,
            "block_offsets": self.block_offsets,
            "D_total": int(D_total),
        }
        with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)

        vocab_pack = {
            "vocab_global": self.vocab_global,
            "idf_global": self.idf_global,
            "index_global": self.index_global,
            "vocab_class": self.vocab_class,
            "idf_class": self.idf_class,
            "index_class": self.index_class,
        }
        with open(os.path.join(out_dir, "vocab.pkl"), "wb") as f:
            pickle.dump(vocab_pack, f)

        # 8) Optional: emit a row-normalized rankingset.pt
        if self.settings.emit_rankingset:
            fps_t = torch.from_numpy(fps)
            norms = torch.linalg.norm(fps_t, dim=1, keepdim=True).clamp_min(1e-12)
            fps_row_norm = (fps_t / norms).to(torch.float32)
            torch.save(fps_row_norm, os.path.join(out_dir, "rankingset.pt"))

    # --------------------
    # Vocab building
    # --------------------
    def _build_class_vocabs_exclusive(
        self,
        df_global: Dict[Feature, int],
        df_class: Dict[str, Dict[Feature, int]],
        N: int,
        class_sizes: Dict[str, int],
    ) -> None:
        if not self.classes:
            self.vocab_class, self.idf_class, self.index_class = {}, {}, {}
            return

        # Compute enrichment for all class-present features
        enrich_by_class: Dict[str, Dict[Feature, float]] = {c: {} for c in self.classes}
        support_by_class: Dict[str, Dict[Feature, int]] = {c: {} for c in self.classes}

        for c in self.classes:
            dfc = df_class[c]
            Nc = max(1, class_sizes[c])
            N_not = max(1, N - Nc)
            for f, dfc_f in dfc.items():
                pc = (dfc_f + self.settings.eps) / (Nc + 2 * self.settings.eps)
                df_not = df_global.get(f, 0) - dfc_f
                pnot = (df_not + self.settings.eps) / (N_not + 2 * self.settings.eps)
                enrich = math.log(max(pc / max(pnot, 1e-12), 1e-12))
                enrich_by_class[c][f] = enrich
                support_by_class[c][f] = dfc_f

        # Argmax assignment: choose best class per feature
        best_class_for_f: Dict[Feature, str] = {}
        best_enrich_for_f: Dict[Feature, float] = {}
        for c, table in enrich_by_class.items():
            for f, e in table.items():
                # apply thresholds
                if support_by_class[c][f] < self.settings.min_support or e < self.settings.enrich_tau:
                    continue
                if (f not in best_enrich_for_f) or (e > best_enrich_for_f[f]):
                    best_enrich_for_f[f] = e
                    best_class_for_f[f] = c

        # Candidate pools per class (exclusive)
        cand_per_class: Dict[str, List[Tuple[Feature, float, int]]] = {c: [] for c in self.classes}
        for f, c in best_class_for_f.items():
            e = best_enrich_for_f[f]
            s = support_by_class[c][f]
            cand_per_class[c].append((f, e, s))

        # Determine K_c per class
        if self.settings.class_vocab_mode == "fixed":
            Kc = {c: self.settings.class_vocab_size for c in self.classes}
        else:
            # sqrt budget allocation with caps
            sizes = np.array([max(1, class_sizes[c]) for c in self.classes], dtype=np.float64)
            weights = np.sqrt(sizes)
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
            raw = weights * float(self.settings.class_vocab_total)
            Kc = {
                c: int(np.clip(int(round(v)), self.settings.class_k_min, self.settings.class_k_max))
                for c, v in zip(self.classes, raw)
            }

        # Select top-Kc by enrichment (tie-break by support)
        rng = np.random.RandomState(self.settings.seed)
        for c in self.classes:
            cand = cand_per_class[c]
            if not cand:
                self.vocab_class[c] = []
                self.idf_class[c] = {}
                self.index_class[c] = {}
                continue
            # Sort: primary by enrichment desc, secondary by support desc
            cand.sort(key=lambda t: (t[1], t[2]), reverse=True)
            kept = [t[0] for t in cand[: Kc[c] ]]
            self.vocab_class[c] = kept
            Nc = max(1, class_sizes[c])
            dfc = df_class[c]
            self.idf_class[c] = {f: math.log((Nc + 1) / (dfc.get(f, 0) + 1)) for f in kept}
            self.index_class[c] = {f: j for j, f in enumerate(kept)}

        # Track the set of all features consumed by classes
        self._class_assigned_feats = set()
        for feats in self.vocab_class.values():
            self._class_assigned_feats.update(feats)

    def _build_global_vocab(self, df_global: Dict[Feature, int], N: int) -> None:
        feats = list(df_global.keys())
        # Optionally remove any class-assigned features from global pool
        if getattr(self, "_class_assigned_feats", None) is not None and self.settings.exclude_class_feats_from_global:
            feats = [f for f in feats if f not in self._class_assigned_feats]
        counts = np.array([df_global[f] for f in feats], dtype=np.float64)
        p = counts / max(1, N)
        entropy = p * np.log2(np.clip(p, 1e-12, 1.0)) + (1 - p) * np.log2(np.clip(1 - p, 1e-12, 1.0))

        if self.settings.global_selection == "entropy":
            order = np.argsort(entropy, kind="stable")  # most negative first (highest entropy)
        else:
            mask = (p >= self.settings.df_min_frac) & (p <= self.settings.df_max_frac)
            idxs = np.where(mask)[0]
            sub = idxs[np.argsort(counts[idxs])[::-1]]
            order = sub

        K = min(self.settings.global_vocab_size, len(order))
        chosen = [feats[i] for i in order[:K]]
        self.vocab_global = chosen
        self.idf_global = {f: math.log((N + 1) / (df_global[f] + 1)) for f in chosen}
        self.index_global = {f: j for j, f in enumerate(chosen)}

    def _compute_block_offsets(self) -> int:
        D = 0
        self.block_offsets["global"] = (D, D + len(self.vocab_global))
        D += len(self.vocab_global)
        for c in self.classes:
            lo, hi = D, D + len(self.vocab_class.get(c, []))
            self.block_offsets[f"class::{c}"] = (lo, hi)
            D = hi
        return D

    # --------------------
    # Vectorization
    # --------------------
    def _vector_global(self, feats: Dict[Feature, int]) -> np.ndarray:
        v = np.zeros(len(self.vocab_global), dtype=np.float32)
        for f, cnt in feats.items():
            j = self.index_global.get(f)
            if j is None:
                continue
            tf = 1.0 if self.settings.tf_mode == "binary" else math.log1p(cnt)
            v[j] = tf * float(self.idf_global[f])
        return v

    def _vector_class_block(self, feats: Dict[Feature, int], cls: str) -> np.ndarray:
        vocab = self.vocab_class.get(cls, [])
        v = np.zeros(len(vocab), dtype=np.float32)
        idx_map = self.index_class.get(cls, {})
        idf_map = self.idf_class.get(cls, {})
        for f, cnt in feats.items():
            j = idx_map.get(f)
            if j is None:
                continue
            tf = 1.0 if self.settings.tf_mode == "binary" else math.log1p(cnt)
            v[j] = tf * float(idf_map[f])
        return v

    @staticmethod
    def _l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        n = float(np.linalg.norm(x))
        return x if n < eps else (x / n)

    # --------------------
    # RDKit helpers
    # --------------------
    def _smiles_to_mol(self, smi: str) -> Optional[Chem.Mol]:
        try:
            return Chem.MolFromSmiles(smi)
        except Exception:
            return None

    def _largest_organic_fragment(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        if rdMolStandardize is None:
            return mol
        try:
            chooser = rdMolStandardize.LargestFragmentChooser()
            mol2 = chooser.choose(mol)
            return mol2
        except Exception:
            return mol

    def _fragment_counts(self, mol: Chem.Mol) -> Dict[Feature, int]:
        _ = self._gen.GetSparseFingerprint(mol, additionalOutput=self._ao)
        info = self._ao.GetBitInfoMap()
        feats: Dict[Feature, int] = {}
        for bit_id, atom_envs in info.items():
            for atom_idx, curr_radius in atom_envs:
                if curr_radius not in self.settings.radii:
                    continue
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, curr_radius, atom_idx)
                submol = Chem.PathToSubmol(mol, env)
                frag_smi = Chem.MolToSmiles(submol, canonical=True)
                atom_symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
                key: Feature = (int(bit_id), str(atom_symbol), str(frag_smi), int(curr_radius))
                feats[key] = feats.get(key, 0) + 1
        return feats


# --------------------
# CLI
# --------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build hybrid TF–IDF fingerprints with exclusive class bits")
    parser.add_argument("input_json", help="JSON path: dict{SMILES: label|[labels]} or list[SMILES]")
    parser.add_argument("out_dir", help="Output directory")

    # TF / radii
    parser.add_argument("--tf_mode", choices=["binary", "log"], default="binary")
    parser.add_argument("--radii", default="1,2,3", help="Comma-separated radii, e.g. 1,2,3")

    # Global vocab
    parser.add_argument("--global_vocab_size", type=int, default=10000)
    parser.add_argument("--global_selection", choices=["entropy", "df_window"], default="entropy")
    parser.add_argument("--df_min_frac", type=float, default=0.01)
    parser.add_argument("--df_max_frac", type=float, default=0.60)
    parser.add_argument("--include_class_feats_in_global", action="store_true",
                        help="If set, class-selected features may also appear in global")

    # Class vocab sizing
    parser.add_argument("--class_vocab_mode", choices=["fixed", "sqrt_budget"], default="sqrt_budget")
    parser.add_argument("--class_vocab_size", type=int, default=256)
    parser.add_argument("--class_vocab_total", type=int, default=6000)
    parser.add_argument("--class_k_min", type=int, default=64)
    parser.add_argument("--class_k_max", type=int, default=512)

    # Enrichment / stats
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--min_support", type=int, default=5)
    parser.add_argument("--enrich_tau", type=float, default=math.log(1.5))

    # Cleanup / normalization / weights
    parser.add_argument("--no_standardize", action="store_true")
    parser.add_argument("--normalize_blocks", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--apply_weights", action="store_true")

    # Misc
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--emit_rankingset", action="store_true",
                        help="Also save row-normalized rankingset.pt in out_dir")

    args = parser.parse_args()

    # Load input
    with open(args.input_json, "r") as f:
        blob = json.load(f)
    if isinstance(blob, dict):
        smiles_and_labels: List[Tuple[str, List[str]]]= []
        for smi, lab in blob.items():
            if lab is None:
                labs = []
            elif isinstance(lab, str):
                labs = [lab]
            elif isinstance(lab, list):
                labs = [str(x) for x in lab]
            else:
                labs = []
            smiles_and_labels.append((smi, labs))
    elif isinstance(blob, list):
        smiles_and_labels = [(smi, []) for smi in blob]
    else:
        raise ValueError("input_json must be dict{SMILES: label|[labels]|null} or list[SMILES]")

    settings = Settings(
        radii=tuple(int(x) for x in args.radii.split(",") if x.strip()),
        tf_mode=args.tf_mode,
        global_vocab_size=args.global_vocab_size,
        global_selection=args.global_selection,
        df_min_frac=args.df_min_frac,
        df_max_frac=args.df_max_frac,
        exclude_class_feats_from_global=(not args.include_class_feats_in_global),
        class_vocab_mode=args.class_vocab_mode,
        class_vocab_size=args.class_vocab_size,
        class_vocab_total=args.class_vocab_total,
        class_k_min=args.class_k_min,
        class_k_max=args.class_k_max,
        eps=args.eps,
        min_support=args.min_support,
        enrich_tau=args.enrich_tau,
        keep_largest_organic_fragment=(not args.no_standardize),
        normalize_blocks=args.normalize_blocks,
        alpha=args.alpha,
        beta=args.beta,
        apply_weights=args.apply_weights,
        seed=args.seed,
        emit_rankingset=args.emit_rankingset,
    )

    builder = IRFPExclusiveBuilder(smiles_and_labels, settings)
    builder.fit_and_build(args.out_dir)
    print(f"Saved IRFP (exclusive class bits) to {args.out_dir}")
