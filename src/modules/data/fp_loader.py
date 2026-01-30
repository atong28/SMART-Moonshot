# fp_loader.py
from __future__ import annotations

import os
import time
import pickle
import argparse
import json
from typing import Optional, Dict, List

import numpy as np
import torch
import lmdb

from ..core.const import DATASET_ROOT
from ..log import get_logger

from .fp_utils import (
    BitInfo as Feature,            # (bit_id, atom_symbol, frag_smiles, radius)
    compute_entropy,
    load_smiles_index,
    count_fragments_over_retrieval,
    write_counts,
    build_rankingset_csr,
    generate_fragments_for_training,
    count_circular_substructures,
)


class _PIDAwareLMDB:
    """
    Read-only, lazily opened LMDB env that re-opens per-process (PID-aware).
    Keys are utf-8 stringified idx; values are raw bytes with header:
      b"{dtype}|{ndim}|{d0},{d1},...|<raw-bytes>"
    """

    def __init__(self, path: str):
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

    def get_bytes(self, key_str: str) -> bytes:
        key = key_str.encode("utf-8")
        env = self._env_for_pid()
        # single retry on stale handle
        try:
            with env.begin(write=False, buffers=True) as txn:
                buf = txn.get(key)
        except lmdb.Error:
            self._env = None
            env = self._env_for_pid()
            with env.begin(write=False, buffers=True) as txn:
                buf = txn.get(key)
        if buf is None:
            raise KeyError(f"Key {key_str} not found in {self.path}")
        return bytes(buf)  # header parse needs a tiny copy of header anyway


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
    Select top-K features by entropy (from retrieval set), then:
      • build MFPs for training molecules using pre-mapped FragIdx (LMDB int32 arrays)
      • build a CSR rankingset over the retrieval molecules (optional)

    Data layout (required):
        DATASET_ROOT/_lmdb/<split>/FragIdx.lmdb        # per-idx sorted int32 col indices
    """

    def __init__(
        self,
        dataset_root: str = DATASET_ROOT,
        retrieval_path: Optional[str] = None,
    ) -> None:
        # Config
        self.dataset_root = dataset_root
        self.retrieval_path = retrieval_path

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
        # store both split keys and FragIdx keys
        self._frag_envs: Dict[str, Optional[_PIDAwareLMDB]] = {}

    # ---------- internal helpers: index & lmdb ----------

    def _ensure_index(self):
        if self._index_loaded:
            return
        index_path = os.path.join(self.dataset_root, "index.pkl")
        with open(index_path, "rb") as f:
            index = pickle.load(f)
        # Build compact idx→split map
        self._idx_to_split = {int(idx): entry.get("split", "train")
                              for idx, entry in index.items()}
        self._index_loaded = True

    def _split_for_idx(self, idx: int) -> str:
        self._ensure_index()
        return self._idx_to_split[int(idx)]  # KeyError → fail fast if missing

    def _ensure_fragidx_env_for_split(self, split: str) -> _PIDAwareLMDB:
        key = f"{split}__FragIdx"
        env = self._frag_envs.get(key)
        if env is None:
            lmdb_path = os.path.join(
                self.dataset_root, "_lmdb", split, "FragIdx.lmdb")
            if not os.path.isdir(lmdb_path):
                raise FileNotFoundError(f"Missing FragIdx shard: {lmdb_path}")
            env = _PIDAwareLMDB(lmdb_path)
            self._frag_envs[key] = env
        return env

    def _load_fragment_indices(self, idx: int) -> np.ndarray:
        """
        Returns np.int32 array of sorted unique FP column indices for this idx.
        """
        split = self._split_for_idx(idx)
        env = self._ensure_fragidx_env_for_split(split)
        buf = env.get_bytes(str(idx))

        # Parse raw header "{dtype}|{ndim}|{d0},{d1},...|" + raw bytes (dtype must be int32)
        mv = memoryview(buf)
        b = mv.tobytes()
        first = b.index(b'|')
        second = b.index(b'|', first + 1)
        third = b.index(b'|', second + 1)
        dtype_str = b[:first].decode('ascii')
        if dtype_str != "int32":
            raise TypeError(f"FragIdx dtype must be int32, got {dtype_str}")
        # ndim/shape in header are ignored for safety; reconstruct via frombuffer
        arr = np.frombuffer(mv[third + 1:], dtype=np.int32)
        return arr

    # ---------- input SMILES iterator (unchanged utility) ----------

    def _iter_smiles_from_path(self, path: str):
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

    def _counts_path(self, radius: int) -> str:
        return os.path.join(self.dataset_root, f"count_hashes_under_radius_{radius}.pkl")

    def prepare_from_retrieval(self, radius: int, num_procs: int = 0) -> None:
        if self.retrieval_path is None:
            raise ValueError("EntropyFPLoader: retrieval_path not set.")
        counts_path = self._counts_path(radius)
        if os.path.exists(counts_path):
            with open(counts_path, "rb") as f:
                self.hashed_bits_count = pickle.load(f)
            return
        counter = count_fragments_over_retrieval(
            self.retrieval_path, radius, num_procs=num_procs)
        write_counts(counter, counts_path)
        self.hashed_bits_count = counter

    def setup(self, out_dim, max_radius, retrieval_path: Optional[str] = None, num_procs: int = 0):
        logger = get_logger(__file__)
        logger.info("Setting up EntropyFPLoader...")
        start = time.time()

        if retrieval_path is not None:
            self.retrieval_path = retrieval_path

        if self.max_radius == max_radius and self.out_dim == out_dim and self.bitinfo_to_fp_index_map:
            logger.info("EntropyFPLoader is already setup.")
            return

        self.max_radius = int(max_radius)

        self.prepare_from_retrieval(
            radius=self.max_radius, num_procs=num_procs)
        if not self.hashed_bits_count:
            raise RuntimeError("Failed to load or compute retrieval counts.")

        filtered = [((bit_id, atom_symbol, frag, r), c)
                    for (bit_id, atom_symbol, frag, r), c in self.hashed_bits_count.items()
                    if r <= self.max_radius]
        if not filtered:
            raise RuntimeError(
                "No features <= max_radius found in retrieval counts.")

        bitinfos, counts = zip(*filtered)
        counts = np.asarray(counts)
        logger.debug(
            f"Found {len(bitinfos)} features with radius <= {self.max_radius}.")

        retrieval_size = len(load_smiles_index(self.retrieval_path))
        ent = compute_entropy(counts, total_dataset_size=retrieval_size)

        k = len(filtered) if (out_dim == "inf" or out_dim ==
                              float("inf")) else int(out_dim)
        topk_idx = np.argpartition(-ent, kth=min(k, len(ent)-1))[:k]
        topk_sorted = sorted(topk_idx, key=lambda i: (-ent[i], bitinfos[i]))

        self.out_dim = len(topk_sorted)
        self.bitinfo_to_fp_index_map = {
            bitinfos[i]: j for j, i in enumerate(topk_sorted)}
        self.fp_index_to_bitinfo_map = {
            v: k for k, v in self.bitinfo_to_fp_index_map.items()}

        elapsed = time.time() - start
        logger.info(
            f"Done! Selected {self.out_dim} features in {elapsed:.2f}s.")

    def build_mfp(self, idx: int) -> torch.Tensor:
        if self.out_dim is None:
            raise RuntimeError("Call setup() first.")
        cols = self._load_fragment_indices(idx)
        mfp = np.zeros(self.out_dim, dtype=np.float32)
        if cols.size > 0:
            mfp[cols] = 1.0
        return torch.from_numpy(mfp)

    def build_mfp_for_smiles(self, smiles: str, ignore_atoms=None) -> torch.Tensor:
        if self.out_dim is None or self.max_radius is None:
            raise RuntimeError("Call setup() first.")
        mfp = np.zeros(self.out_dim, dtype=np.float32)
        present = count_circular_substructures(
            smiles, radius=self.max_radius, ignore_atoms=ignore_atoms or [])
        for bitinfo in present.keys():
            col = self.bitinfo_to_fp_index_map.get(bitinfo)
            if col is not None and 0 <= col < self.out_dim:
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
                if col is not None and 0 <= col < self.out_dim:
                    mfp[col] = 1.0
        return torch.from_numpy(mfp)

    def build_fp_indices_for_smiles(self, smiles: str, ignore_atoms=None) -> Optional[List[int]]:
        if self.out_dim is None or self.max_radius is None:
            raise RuntimeError("Call setup() first.")
        present = count_circular_substructures(
            smiles, radius=self.max_radius, ignore_atoms=ignore_atoms or [])
        if not present:
            return []
        cols = set()
        for bitinfo in present.keys():
            col = self.bitinfo_to_fp_index_map.get(bitinfo)
            if col is not None and 0 <= col < self.out_dim:
                cols.add(col)
        return sorted(cols)

    def build_fp_dict_for_smiles(self, smiles_list, ignore_atoms=None) -> Dict[str, Optional[List[int]]]:
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
            indices = self.build_fp_indices_for_smiles(
                smi, ignore_atoms=ignore_atoms)
            out[smi] = indices
        return out

    def build_rankingset(self, fp_type: str = "RankingEntropy", save: bool = True, num_procs: int = 0) -> torch.Tensor:
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
            with open(os.path.join(out_dir, "bitinfo_to_idx.pkl"), "wb") as f:
                pickle.dump(self.bitinfo_to_fp_index_map, f)
        return csr

    def load_rankingset(self, fp_type: str):
        rankingset_path = os.path.join(
            self.dataset_root, fp_type, "rankingset.pt")
        return torch.load(rankingset_path, weights_only=True)


def make_fp_loader(fp_type: str, entropy_out_dim=16384, max_radius=6, retrieval_path: Optional[str] = None):
    """_summary_

    Args:
        fp_type (str): _description_
        entropy_out_dim (int, optional): _description_. Defaults to 16384.
        max_radius (int, optional): _description_. Defaults to 6.
        retrieval_path (Optional[str], optional): _description_. Defaults to None.

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """
    if fp_type == "RankingEntropy":
        fp_loader = EntropyFPLoader(retrieval_path=retrieval_path)
        fp_loader.setup(entropy_out_dim, max_radius,
                        retrieval_path=retrieval_path)
        return fp_loader
    raise NotImplementedError(f"FP type {fp_type} not implemented")


def _cli():
    parser = argparse.ArgumentParser(
        description="Fingerprint loader utilities (entropy selection + rankingset builder).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_counts = sub.add_parser(
        "prepare-counts", help="Precompute fragment presence counts on retrieval set.")
    p_counts.add_argument("--retrieval", required=True,
                          help="Path to retrieval index (.pkl/.json) with smiles.")
    p_counts.add_argument("--dataset-root", default=DATASET_ROOT,
                          help="Dataset root where counts file will be stored.")
    p_counts.add_argument("--radius", type=int, default=6)
    p_counts.add_argument("--num-procs", type=int, default=0,
                          help="0=auto, else explicit number.")

    p_rank = sub.add_parser(
        "rankingset", help="Run entropy feature selection and build/save CSR rankingset.")
    p_rank.add_argument("--retrieval", required=True,
                        help="Path to retrieval index (.pkl/.json) with smiles.")
    p_rank.add_argument("--dataset-root", default=DATASET_ROOT)
    p_rank.add_argument("--out-dim", default=16384, help="int or 'inf'")
    p_rank.add_argument("--radius", type=int, default=6)
    p_rank.add_argument("--fp-type", default="RankingEntropy",
                        help="Subdir under dataset root to save artifacts.")
    p_rank.add_argument("--num-procs", type=int, default=0)
    p_rank.add_argument("--no-save", action="store_true",
                        help="If set, do not save rankingset to disk.")

    p_frag = sub.add_parser(
        "fragments", help="Generate per-idx fragment lists for training set.")
    p_frag.add_argument("--index", required=True,
                        help="Path to training index (.pkl/.json) with smiles.")
    p_frag.add_argument("--out-dir", required=True,
                        help="Output directory (will create 'Fragments/' inside).")
    p_frag.add_argument("--radius", type=int, default=6)
    p_frag.add_argument("--num-procs", type=int, default=0)

    p_sfps = sub.add_parser(
        "smiles-fps", help="Build {smiles: List[int]|None} indices and save to a pickle.")
    p_sfps.add_argument("--smiles", required=True,
                        help="Path to txt/json/pkl/csv with SMILES.")
    p_sfps.add_argument("--out", required=True,
                        help="Output pickle path (will contain {smiles: List[int] or None}).")
    p_sfps.add_argument("--retrieval", required=True,
                        help="Path to retrieval index (.pkl/.json) used for feature selection.")
    p_sfps.add_argument("--dataset-root", default=DATASET_ROOT)
    p_sfps.add_argument("--out-dim", default=16384, help="int or 'inf'")
    p_sfps.add_argument("--radius", type=int, default=6)
    p_sfps.add_argument("--ignore-atoms", default="",
                        help="Comma-separated atom indices to ignore (rare; usually leave empty).")
    p_sfps.add_argument("--num-procs", type=int, default=0,
                        help="For counts/CSR prep if needed.")

    args = parser.parse_args()

    if args.cmd == "prepare-counts":
        loader = EntropyFPLoader(
            dataset_root=args.dataset_root, retrieval_path=args.retrieval)
        loader.prepare_from_retrieval(
            radius=args.radius, num_procs=args.num_procs)
        logger = get_logger(__file__)
        logger.info(f"Counts written to {loader._counts_path(args.radius)}")

    elif args.cmd == "rankingset":
        out_dim = args.out_dim
        if isinstance(out_dim, str) and out_dim.lower() == "inf":
            out_dim = "inf"
        else:
            out_dim = int(out_dim)

        loader = EntropyFPLoader(
            dataset_root=args.dataset_root, retrieval_path=args.retrieval)
        loader.setup(out_dim, args.radius,
                     retrieval_path=args.retrieval, num_procs=args.num_procs)
        csr = loader.build_rankingset(fp_type=args.fp_type, save=(
            not args.no_save), num_procs=args.num_procs)
        logger = get_logger(__file__)
        logger.info(f"CSR shape: {tuple(csr.shape)}")
        if not args.no_save:
            logger.info(
                f"Saved rankingset to {os.path.join(args.dataset_root, args.fp_type, 'rankingset.pt')}")

    elif args.cmd == "fragments":
        generate_fragments_for_training(
            index_path=args.index, out_dir=args.out_dir, radius=args.radius, num_procs=args.num_procs
        )
        logger = get_logger(__file__)
        logger.info(
            f"Fragments written under {os.path.join(args.out_dir, 'Fragments')}")

    elif args.cmd == "smiles-fps":
        out_dim = args.out_dim
        if isinstance(out_dim, str) and out_dim.lower() == "inf":
            out_dim = "inf"
        else:
            out_dim = int(out_dim)

        ignore_atoms = []
        if args.ignore_atoms.strip():
            ignore_atoms = [int(x) for x in args.ignore_atoms.split(
                ",") if x.strip().isdigit()]

        loader = EntropyFPLoader(
            dataset_root=args.dataset_root, retrieval_path=args.retrieval)
        loader.setup(out_dim, args.radius,
                     retrieval_path=args.retrieval, num_procs=args.num_procs)

        smiles_iter = loader._iter_smiles_from_path(args.smiles)
        fp_dict = loader.build_fp_dict_for_smiles(
            smiles_iter, ignore_atoms=ignore_atoms)

        out_dir = os.path.dirname(os.path.abspath(args.out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "wb") as f:
            pickle.dump(fp_dict, f)

        n_total = len(fp_dict)
        n_none = sum(1 for v in fp_dict.values() if v is None)
        logger = get_logger(__file__)
        logger.info(
            f"Wrote {n_total} entries to {args.out} ({n_none} parse failures).")


if __name__ == "__main__":
    _cli()
