# fp_loader.py
from __future__ import annotations

import os
import io
import time
import pickle
import argparse
import json
from typing import Optional, Dict, List, Union

import numpy as np
import torch

try:
    import lmdb
except ImportError:
    lmdb = None

from ..core.const import DATASET_ROOT, CODE_ROOT
from .fp_utils import (
    BitInfo as Feature,            # (bit_id, atom_symbol, frag_smiles, radius)
    compute_entropy,
    load_smiles_index,
    count_fragments_over_retrieval,
    write_counts,
    build_rankingset_csr,
    generate_fragments_for_training,
)

# ----------------------------------------------------------------------
# Internal: PID-aware read-only LMDB wrapper (lazy-open per process)
# ----------------------------------------------------------------------
class _PIDAwareLMDB:
    """
    Read-only, lazily opened LMDB env that re-opens per-process (PID-aware).
    Keys are utf-8 stringified idx; values are torch.save()-serialized bytes.
    """
    def __init__(self, path: str):
        if lmdb is None:
            raise RuntimeError("lmdb is not installed. `pip install lmdb`")
        self.path = path
        self._env = None
        self._pid = None

    def _env_for_pid(self):
        pid = os.getpid()
        if self._env is None or self._pid != pid:
            self._env = lmdb.open(
                self.path,
                readonly=True,
                lock=False,
                readahead=False,
                subdir=True,
                max_readers=4096,
            )
            self._pid = pid
        return self._env

    def get_bytes(self, key_str: str) -> Optional[bytes]:
        key = key_str.encode("utf-8")
        env = self._env_for_pid()
        try:
            with env.begin(write=False) as txn:
                buf = txn.get(key)
        except Exception:
            # stale / forked handle → reopen once
            self._env = None
            env = self._env_for_pid()
            with env.begin(write=False) as txn:
                buf = txn.get(key)
        return buf

# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
class FPLoader:
    def __init__(self) -> None:
        raise NotImplementedError()

    def setup(self, out_dim, max_radius, **kwargs):
        raise NotImplementedError()

    def build_mfp(self, idx: int) -> torch.Tensor:
        raise NotImplementedError()

    def load_rankingset(self, fp_type: str):
        raise NotImplementedError()


class EntropyFPLoader(FPLoader):
    """
    Selects features by (positive) entropy estimated on the retrieval set, then
    produces MFPS for training molecules and a CSR rankingset for the retrieval molecules.

    Key addition: single loader instance that auto-routes idx→split and reads
    Fragments from split-scoped LMDB shards if present:
        DATASET_ROOT/_lmdb/<split>/Fragments.lmdb
    Fallback: DATASET_ROOT/Fragments/{idx}.pt
    """

    def __init__(
        self,
        dataset_root: str = DATASET_ROOT,
        retrieval_path: Optional[str] = None,
    ) -> None:
        # Config
        self.dataset_root = dataset_root
        self.retrieval_path = retrieval_path  # path to retrieval.json / retrieval.pkl

        # Selection state
        self.hashed_bits_count: Optional[Dict[Feature, int]] = None
        self.max_radius: Optional[int] = None
        self.out_dim: Optional[int] = None
        self.bitinfo_to_fp_index_map: Dict[Feature, int] = {}
        self.fp_index_to_bitinfo_map: Dict[int, Feature] = {}

        # Index for idx→split routing (loaded lazily)
        self._idx_to_split: Optional[Dict[int, str]] = None
        self._index_loaded: bool = False

        # Split-scoped LMDB envs (lazy-open on first use)
        # Keys: "train" | "val" | "test" ; Values: _PIDAwareLMDB or None if missing
        self._frag_envs: Dict[str, Optional[_PIDAwareLMDB]] = {
            "train": None, "val": None, "test": None
        }
        self._probed_envs: Dict[str, bool] = {
            "train": False, "val": False, "test": False
        }

    # ---------- internal helpers: index & lmdb ----------

    def _ensure_index(self):
        if self._index_loaded:
            return
        index_path = os.path.join(self.dataset_root, "index.pkl")
        with open(index_path, "rb") as f:
            index = pickle.load(f)
        # Build compact idx→split map
        self._idx_to_split = {int(idx): entry.get("split", "train") for idx, entry in index.items()}
        self._index_loaded = True

    def _split_for_idx(self, idx: int) -> Optional[str]:
        self._ensure_index()
        # type: ignore[union-attr]
        return self._idx_to_split.get(int(idx)) if self._idx_to_split is not None else None

    def _ensure_frag_env_for_split(self, split: str) -> Optional[_PIDAwareLMDB]:
        if self._probed_envs.get(split, False):
            return self._frag_envs.get(split)
        # First time we see this split: probe shard path
        lmdb_path = os.path.join(self.dataset_root, "_lmdb", split, "Fragments.lmdb")
        if os.path.isdir(lmdb_path):
            try:
                env = _PIDAwareLMDB(lmdb_path)
            except Exception:
                env = None
        else:
            env = None
        self._frag_envs[split] = env
        self._probed_envs[split] = True
        return env

    def _load_fragments(self, idx: int):
        """
        Returns the deserialized object stored in Fragments for `idx`
        (usually a list[BitInfo]). Tries LMDB shard first, then .pt file.
        """
        split = self._split_for_idx(idx)
        if split is not None:
            env = self._ensure_frag_env_for_split(split)
            if env is not None:
                buf = env.get_bytes(str(idx))
                if buf is not None:
                    return torch.load(io.BytesIO(buf), map_location="cpu")

        # Fallback to filesystem
        filepath = os.path.join(self.dataset_root, "Fragments", f"{idx}.pt")
        return torch.load(filepath, weights_only=True)

    # ---------- input SMILES iterator ----------

    def _iter_smiles_from_path(self, path: str):
        """
        Yields SMILES strings from a file:
          - .txt: one SMILES per non-empty line
          - .json: list[str], or {'smiles': [...]}, or dict of id->smiles (values used)
          - .pkl/.pickle: same as .json (list[str] / dict / {'smiles': [...]})
          - .csv: uses 'smiles' column if present; else first column
        """
        ext = os.path.splitext(path)[1].lower()
        if ext in {".txt"}:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        yield s
            return

        if ext in {".json"}:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                for s in obj:
                    if isinstance(s, str) and s:
                        yield s
            elif isinstance(obj, dict):
                if "smiles" in obj and isinstance(obj["smiles"], list):
                    for s in obj["smiles"]:
                        if isinstance(s, str) and s:
                            yield s
                else:
                    for v in obj.values():
                        if isinstance(v, str) and v:
                            yield v
            else:
                raise ValueError(f"Unrecognized JSON structure in {path}")
            return

        if ext in {".pkl", ".pickle"}:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, list):
                for s in obj:
                    if isinstance(s, str) and s:
                        yield s
            elif isinstance(obj, dict):
                if "smiles" in obj and isinstance(obj["smiles"], list):
                    for s in obj["smiles"]:
                        if isinstance(s, str) and s:
                            yield s
                else:
                    for v in obj.values():
                        if isinstance(v, str) and v:
                            yield v
            else:
                raise ValueError(f"Unrecognized PKL structure in {path}")
            return

        if ext in {".csv"}:
            with open(path, "r", encoding="utf-8") as f:
                header = f.readline()
                if not header:
                    return
                cols = [c.strip() for c in header.strip().split(",")]
                try:
                    smiles_idx = cols.index("smiles")
                except ValueError:
                    smiles_idx = 0  # fall back to first column
                for line in f:
                    if not line.strip():
                        continue
                    parts = line.rstrip("\n").split(",")
                    if smiles_idx < len(parts):
                        s = parts[smiles_idx].strip()
                        if s:
                            yield s
            return

        raise ValueError(f"Unsupported input format for {path}")

    # ---------- retrieval prep ----------

    def _counts_path(self, radius: int) -> str:
        return os.path.join(self.dataset_root, f"count_hashes_under_radius_{radius}.pkl")

    def prepare_from_retrieval(self, radius: int, num_procs: int = 0) -> None:
        """
        Ensure we have retrieval counts on disk; if not, compute from retrieval_path.
        """
        if self.retrieval_path is None:
            raise ValueError("EntropyFPLoader: retrieval_path not set.")
        counts_path = self._counts_path(radius)
        if os.path.exists(counts_path):
            with open(counts_path, "rb") as f:
                self.hashed_bits_count = pickle.load(f)
            return
        # compute & write
        counter = count_fragments_over_retrieval(self.retrieval_path, radius, num_procs=num_procs)
        write_counts(counter, counts_path)
        self.hashed_bits_count = counter

    # ---------- setup (feature selection) ----------

    def setup(self, out_dim, max_radius, retrieval_path: Optional[str] = None, num_procs: int = 0):
        """
        Select top-K features by entropy on the retrieval set.
        - out_dim: int or 'inf' to use all available features <= radius.
        - max_radius: Morgan radius.
        - retrieval_path: optional override.
        """
        print("Setting up EntropyFPLoader...")
        start = time.time()

        if retrieval_path is not None:
            self.retrieval_path = retrieval_path

        if self.max_radius == max_radius and self.out_dim == out_dim and self.bitinfo_to_fp_index_map:
            print("EntropyFPLoader is already setup.")
            return

        self.max_radius = int(max_radius)

        self.prepare_from_retrieval(radius=self.max_radius, num_procs=num_procs)
        if not self.hashed_bits_count:
            raise RuntimeError("Failed to load or compute retrieval counts.")

        # filter by radius
        filtered = [((bit_id, atom_symbol, frag, r), c)
                    for (bit_id, atom_symbol, frag, r), c in self.hashed_bits_count.items()
                    if r <= self.max_radius]
        if not filtered:
            raise RuntimeError("No features <= max_radius found in retrieval counts.")

        bitinfos, counts = zip(*filtered)
        counts = np.asarray(counts)
        print(f"Found {len(bitinfos)} features with radius <= {self.max_radius}.")

        # retrieval size = #smiles in retrieval index
        retrieval_size = len(load_smiles_index(self.retrieval_path))

        # positive entropy, pick largest entropies
        ent = compute_entropy(counts, total_dataset_size=retrieval_size)

        if out_dim == "inf" or out_dim == float("inf"):
            k = len(filtered)
        else:
            k = int(out_dim)

        topk_idx = np.argpartition(-ent, kth=min(k, len(ent)-1))[:k]
        # stable order by entropy (desc), then deterministic tiebreaker by tuple
        topk_sorted = sorted(topk_idx, key=lambda i: (-ent[i], bitinfos[i]))

        self.out_dim = len(topk_sorted)
        self.bitinfo_to_fp_index_map = {bitinfos[i]: j for j, i in enumerate(topk_sorted)}
        self.fp_index_to_bitinfo_map = {v: k for k, v in self.bitinfo_to_fp_index_map.items()}

        elapsed = time.time() - start
        print(f"Done! Selected {self.out_dim} features in {elapsed:.2f}s.")

    # ---------- per-sample / new SMILES ----------

    def build_mfp(self, idx: int) -> torch.Tensor:
        """
        Build MFP for a training molecule from Fragments:
          1) Try _lmdb/<split>/Fragments.lmdb (split inferred from index.pkl)
          2) Fallback to Fragments/{idx}.pt
        """
        if self.out_dim is None:
            raise RuntimeError("Call setup() first.")

        fragment_infos = self._load_fragments(idx)

        mfp = np.zeros(self.out_dim, dtype=np.float32)
        for frag in fragment_infos:
            col = self.bitinfo_to_fp_index_map.get(frag)
            if col is not None:
                mfp[col] = 1.0
        return torch.from_numpy(mfp)

    def build_mfp_for_smiles(self, smiles: str, ignore_atoms=None) -> torch.Tensor:
        # Legacy dense tensor builder (kept for compatibility)
        from .fp_utils import count_circular_substructures  # local import to avoid cycles
        if self.out_dim is None or self.max_radius is None:
            raise RuntimeError("Call setup() first.")
        mfp = np.zeros(self.out_dim, dtype=np.float32)
        present = count_circular_substructures(smiles, radius=self.max_radius, ignore_atoms=ignore_atoms or [])
        for bitinfo in present.keys():
            col = self.bitinfo_to_fp_index_map.get(bitinfo)
            if col is not None:
                mfp[col] = 1.0
        return torch.from_numpy(mfp)

    def build_mfp_from_bitinfo(self, atom_to_bitinfos: Dict[int, list], ignore_atoms=None) -> torch.Tensor:
        if self.out_dim is None:
            raise RuntimeError("Call setup() first.")
        mfp = np.zeros(self.out_dim, dtype=np.float32)
        ignore = set(ignore_atoms or [])
        for atom_idx, bitinfos in atom_to_bitinfos.items():
            if atom_idx in ignore:
                continue
            for bitinfo in bitinfos:
                col = self.bitinfo_to_fp_index_map.get(bitinfo)
                if col is not None:
                    mfp[col] = 1.0
        return torch.from_numpy(mfp)

    # --- NEW: sparse indices helpers ---

    def build_fp_indices_for_smiles(self, smiles: str, ignore_atoms=None) -> Optional[List[int]]:
        """
        Returns sorted list of 0-indexed bit positions that are 1 for this SMILES.
        If parsing fails or any error occurs, returns None.
        """
        from .fp_utils import count_circular_substructures  # local import to avoid cycles
        if self.out_dim is None or self.max_radius is None:
            raise RuntimeError("Call setup() first.")
        try:
            present = count_circular_substructures(smiles, radius=self.max_radius, ignore_atoms=ignore_atoms or [])
        except Exception:
            return None
        if not present:
            return []
        cols = set()
        for bitinfo in present.keys():
            col = self.bitinfo_to_fp_index_map.get(bitinfo)
            if col is not None:
                cols.add(col)
        return sorted(cols)

    def build_fp_dict_for_smiles(self, smiles_list, ignore_atoms=None) -> Dict[str, Optional[List[int]]]:
        """
        Given an iterable of SMILES strings, returns {smiles: List[int] | None}
        where List[int] are the 0-indexed bit positions set to 1. If parsing fails,
        the value is None. Deduplicates identical SMILES.
        """
        if self.out_dim is None or self.max_radius is None:
            raise RuntimeError("Call setup() first.")

        ignore_atoms = ignore_atoms or []
        seen = set()
        out: Dict[str, Optional[List[int]]] = {}
        for smi in smiles_list:
            if not isinstance(smi, str) or not smi:
                continue
            if smi in seen:
                continue
            seen.add(smi)
            indices = self.build_fp_indices_for_smiles(smi, ignore_atoms=ignore_atoms)
            out[smi] = indices  # may be None
        return out

    # ---------- retrieval rankingset ----------

    def build_rankingset(self, fp_type: str = "RankingEntropy", save: bool = True, num_procs: int = 0) -> torch.Tensor:
        """
        Build torch.sparse_csr_tensor over the retrieval set and (optionally) save to:
            DATASET_ROOT/{fp_type}/rankingset.pt
        """
        if self.max_radius is None or not self.bitinfo_to_fp_index_map:
            raise RuntimeError("Call setup() first.")

        csr = build_rankingset_csr(
            retrieval_path=self.retrieval_path,
            bitinfo_to_col=self.bitinfo_to_fp_index_map,
            radius=self.max_radius,
            num_procs=num_procs,
        )

        if save:
            out_dir = os.path.join(self.dataset_root, fp_type)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "rankingset.pt")
            torch.save(csr, out_path)
            # (optional) persist the mapping so later runs can inspect it
            with open(os.path.join(out_dir, "bitinfo_to_idx.pkl"), "wb") as f:
                pickle.dump(self.bitinfo_to_fp_index_map, f)
        return csr

    def load_rankingset(self, fp_type: str):
        rankingset_path = os.path.join(self.dataset_root, fp_type, "rankingset.pt")
        return torch.load(rankingset_path, weights_only=True)


def make_fp_loader(fp_type: str, entropy_out_dim=16384, max_radius=6, retrieval_path: Optional[str] = None):
    if fp_type == "RankingEntropy":
        fp_loader = EntropyFPLoader(retrieval_path=retrieval_path)
        fp_loader.setup(entropy_out_dim, max_radius, retrieval_path=retrieval_path)
        return fp_loader
    raise NotImplementedError(f"FP type {fp_type} not implemented")


# ---------------------------
# CLI
# ---------------------------
def _cli():
    parser = argparse.ArgumentParser(description="Fingerprint loader utilities (entropy selection + rankingset builder).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # 1) Precompute retrieval counts
    p_counts = sub.add_parser("prepare-counts", help="Precompute fragment presence counts on retrieval set.")
    p_counts.add_argument("--retrieval", required=True, help="Path to retrieval index (.pkl/.json) with smiles.")
    p_counts.add_argument("--dataset-root", default=DATASET_ROOT, help="Dataset root where counts file will be stored.")
    p_counts.add_argument("--radius", type=int, default=6)
    p_counts.add_argument("--num-procs", type=int, default=0, help="0=auto, else explicit number.")

    # 2) Build rankingset (runs selection + CSR)
    p_rank = sub.add_parser("rankingset", help="Run entropy feature selection and build/save CSR rankingset.")
    p_rank.add_argument("--retrieval", required=True, help="Path to retrieval index (.pkl/.json) with smiles.")
    p_rank.add_argument("--dataset-root", default=DATASET_ROOT)
    p_rank.add_argument("--out-dim", default=16384, help="int or 'inf'")
    p_rank.add_argument("--radius", type=int, default=6)
    p_rank.add_argument("--fp-type", default="RankingEntropy", help="Subdir under dataset root to save artifacts.")
    p_rank.add_argument("--num-procs", type=int, default=0)
    p_rank.add_argument("--no-save", action="store_true", help="If set, do not save rankingset to disk.")

    # 3) Generate training fragments
    p_frag = sub.add_parser("fragments", help="Generate per-idx fragment lists for training set.")
    p_frag.add_argument("--index", required=True, help="Path to training index (.pkl/.json) with smiles.")
    p_frag.add_argument("--out-dir", required=True, help="Output directory (will create 'Fragments/' inside).")
    p_frag.add_argument("--radius", type=int, default=6)
    p_frag.add_argument("--num-procs", type=int, default=0)

    # 4) Build FP dict for a list of SMILES (stores indices; parse errors → None)
    p_sfps = sub.add_parser("smiles-fps", help="Build {smiles: List[int]|None} of 1-bit indices and save to a pickle.")
    p_sfps.add_argument("--smiles", required=True, help="Path to txt/json/pkl/csv with SMILES.")
    p_sfps.add_argument("--out", required=True, help="Output pickle path (will contain {smiles: List[int] or None}).")
    p_sfps.add_argument("--retrieval", required=True, help="Path to retrieval index (.pkl/.json) used for feature selection.")
    p_sfps.add_argument("--dataset-root", default=DATASET_ROOT)
    p_sfps.add_argument("--out-dim", default=16384, help="int or 'inf'")
    p_sfps.add_argument("--radius", type=int, default=6)
    p_sfps.add_argument("--ignore-atoms", default="", help="Comma-separated atom indices to ignore (rare; usually leave empty).")
    p_sfps.add_argument("--num-procs", type=int, default=0, help="For counts/CSR prep if needed.")

    args = parser.parse_args()

    if args.cmd == "prepare-counts":
        loader = EntropyFPLoader(dataset_root=args.dataset_root, retrieval_path=args.retrieval)
        loader.prepare_from_retrieval(radius=args.radius, num_procs=args.num_procs)
        print(f"Counts written to {loader._counts_path(args.radius)}")

    elif args.cmd == "rankingset":
        out_dim = args.out_dim
        if isinstance(out_dim, str) and out_dim.lower() == "inf":
            out_dim = "inf"
        else:
            out_dim = int(out_dim)

        loader = EntropyFPLoader(dataset_root=args.dataset_root, retrieval_path=args.retrieval)
        loader.setup(out_dim, args.radius, retrieval_path=args.retrieval, num_procs=args.num_procs)
        csr = loader.build_rankingset(fp_type=args.fp_type, save=(not args.no_save), num_procs=args.num_procs)
        print(f"CSR shape: {tuple(csr.shape)}")
        if not args.no_save:
            print(f"Saved rankingset to {os.path.join(args.dataset_root, args.fp_type, 'rankingset.pt')}")

    elif args.cmd == "fragments":
        generate_fragments_for_training(
            index_path=args.index, out_dir=args.out_dir, radius=args.radius, num_procs=args.num_procs
        )
        print(f"Fragments written under {os.path.join(args.out_dir, 'Fragments')}")

    elif args.cmd == "smiles-fps":
        out_dim = args.out_dim
        if isinstance(out_dim, str) and out_dim.lower() == "inf":
            out_dim = "inf"
        else:
            out_dim = int(out_dim)

        ignore_atoms = []
        if args.ignore_atoms.strip():
            ignore_atoms = [int(x) for x in args.ignore_atoms.split(",") if x.strip().isdigit()]

        loader = EntropyFPLoader(dataset_root=args.dataset_root, retrieval_path=args.retrieval)
        # Ensures selection is available (computes counts if missing)
        loader.setup(out_dim, args.radius, retrieval_path=args.retrieval, num_procs=args.num_procs)

        smiles_iter = loader._iter_smiles_from_path(args.smiles)
        fp_dict = loader.build_fp_dict_for_smiles(smiles_iter, ignore_atoms=ignore_atoms)

        # Save as a pickle of {smiles: List[int] | None}
        out_dir = os.path.dirname(os.path.abspath(args.out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "wb") as f:
            pickle.dump(fp_dict, f)

        # Simple summary
        n_total = len(fp_dict)
        n_none = sum(1 for v in fp_dict.values() if v is None)
        print(f"Wrote {n_total} entries to {args.out} ({n_none} parse failures).")


if __name__ == "__main__":
    _cli()
