# fp_loader.py
from __future__ import annotations

import os
import io
import time
import pickle
import argparse
from typing import Optional, Dict

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


if __name__ == "__main__":
    _cli()
