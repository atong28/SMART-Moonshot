import pickle
import numpy as np
import torch
import os
import json
import time
from typing import Tuple, Optional

from ..core.const import DATASET_ROOT, CODE_ROOT
from .fp_utils import compute_entropy, count_circular_substructures

Feature = Tuple[int, str, str, int]  # (bit_id, atom_symbol, frag_smiles, radius)

class FPLoader:
    def __init__(self) -> None:
        raise NotImplementedError()
    
    def setup(self, out_dim, max_radius):
        raise NotImplementedError() 
    
    def build_mfp(self, idx: int) -> torch.Tensor:
        raise NotImplementedError()
    
    def load_rankingset(self, fp_type: str):
        raise NotImplementedError()

class EntropyFPLoader(FPLoader):
    def __init__(self) -> None:
        self.data_root = DATASET_ROOT
        save_path = os.path.join(self.data_root, "count_hashes_under_radius_10.pkl")
        with open(save_path, "rb") as f:
            self.hashed_bits_count = pickle.load(f)
        self.max_radius = None
        self.out_dim = None

    def build_rankingset(self, split):   
        # TODO: fix this path/calculation
        # assuming rankingset on allinfo-set
        path_to_load_full_info_indices = f"{CODE_ROOT}/datasets/{split}_indices_of_full_info_NMRs.pkl"
        file_idx_for_ranking_set = pickle.load(open(path_to_load_full_info_indices, "rb"))

        files  = [self.build_mfp(int(file_idx.split(".")[0]), "2d", split) for file_idx in sorted(file_idx_for_ranking_set)]
        out = torch.vstack(files)
        return out
        
    def setup(self, out_dim, max_radius):
        print('Setting up EntropyFPLoader...')
        start = time.time()
        if self.out_dim == out_dim and self.max_radius == max_radius:
            print("EntropyFPLoader is already setup")
            return

        self.max_radius = max_radius
        filtered_bitinfos_and_their_counts = [((bit_id, atom_symbol, frag_smiles, radius), counts)  for (bit_id, atom_symbol, frag_smiles, radius), counts in self.hashed_bits_count.items() if radius <= max_radius]
        bitinfos, counts = zip(*filtered_bitinfos_and_their_counts)
        counts = np.array(counts)
        if out_dim == 'inf' or out_dim == float("inf"):
            out_dim = len(filtered_bitinfos_and_their_counts)
        self.out_dim = out_dim
        retrieval_set_size = 526316
        entropy_each_frag = compute_entropy(counts, total_dataset_size = retrieval_set_size)
        indices_of_high_entropy = np.argsort(entropy_each_frag, kind="stable")[:out_dim]
        self.bitInfos_to_fp_index_map = {bitinfos[bitinfo_list_index]: fp_index for fp_index, bitinfo_list_index in enumerate(indices_of_high_entropy)}
        self.fp_index_to_bitInfo_mapping =  {v:k for k, v in self.bitInfos_to_fp_index_map.items()}
        end = time.time()
        print(f'Done! Took {end-start} seconds')
    def build_mfp(self, idx):
        filepath = os.path.join(self.data_root, 'Fragments', f'{idx}.pt')
        fragment_infos = torch.load(filepath, weights_only=True) 
        mfp = np.zeros(self.out_dim)
        for frag_info in fragment_infos:
            if frag_info in self.bitInfos_to_fp_index_map:
                mfp[self.bitInfos_to_fp_index_map[frag_info]] = 1
        return torch.tensor(mfp).float()

    def build_mfp_for_new_SMILES(self, smiles, ignoreAtoms = []):
        mfp = np.zeros(self.out_dim)
        
        bitInfos_with_count = count_circular_substructures(smiles, ignoreAtoms = ignoreAtoms)
        for bitInfo in bitInfos_with_count:
            if bitInfo in self.bitInfos_to_fp_index_map:
                mfp[self.bitInfos_to_fp_index_map[bitInfo]] = 1
        return torch.tensor(mfp).float()
    
    def build_mfp_from_bitInfo(self, atom_to_bitInfos, ignoreAtoms = []):
        # atom_to_bitInfos: a dict of atom index to bitInfo
        mfp = np.zeros(self.out_dim)
        for atom_idx, bitInfos in atom_to_bitInfos.items():
            if atom_idx in ignoreAtoms:
                continue
            for bitInfo in bitInfos:
                if bitInfo in self.bitInfos_to_fp_index_map:
                    mfp[self.bitInfos_to_fp_index_map[bitInfo]] = 1
        return torch.tensor(mfp).float()
    
    def load_rankingset(self, fp_type: str):
        rankingset_path = os.path.join(DATASET_ROOT, fp_type, 'rankingset.pt')
        return torch.load(rankingset_path, weights_only=True)

# data/fp_loader.py  (add this class)

class IRFPFPLoader:
    """
    Loads precomputed TF-IDF fingerprints saved per-idx at:
      DATASET_ROOT/{fp_type}/fp/{idx}.pt  (float32)

    Also exposes load_rankingset(fp_type) that returns the row-normalized bank.
    """
    def __init__(self, fp_type: str):
        self.fp_type = fp_type
        self.root = os.path.join(DATASET_ROOT, fp_type)
        self.fp_dir = os.path.join(self.root, "fp")
        os.makedirs(self.fp_dir, exist_ok=True)

    def build_mfp(self, idx: int) -> torch.Tensor:
        """
        Return the (D,) float32 TF-IDF vector for this idx, as saved by materializer.
        """
        path = os.path.join(self.fp_dir, f"{idx}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing per-idx fp: {path}. Run materialize_irfp_per_idx.py.")
        return torch.load(path, weights_only=True).to(torch.float32)

    def load_rankingset(self, fp_type: Optional[str] = None) -> torch.Tensor:
        """
        Row-normalized (N, D) tensor for fast cosine ranking.
        Falls back to fingerprints.pt + on-the-fly row L2 if rankingset.pt is missing.
        """
        root = self.root if fp_type is None else os.path.join(DATASET_ROOT, fp_type)
        rs_path = os.path.join(root, "rankingset.pt")
        if os.path.exists(rs_path):
            return torch.load(rs_path, weights_only=True).to(torch.float32)
        # Fallback
        fps = torch.load(os.path.join(root, "fingerprints.pt"), weights_only=True).to(torch.float32)
        norms = torch.linalg.norm(fps, dim=1, keepdim=True).clamp_min(1e-12)
        return fps / norms

    # Optional helpers
    def get_out_dim(self) -> int:
        with open(os.path.join(self.root, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        return int(meta["D_total"])

    def smiles_to_idx(self) -> dict:
        p = os.path.join(self.root, "smiles_to_idx.json")
        return json.load(open(p)) if os.path.exists(p) else {}
    
class BiosynfoniFPLoader(FPLoader):
    """
    Loads 39-D Biosynfoni (log1p count) fingerprints.

    Expects files under DATASET_ROOT/Biosynfoni:
      - rankingset.pt         (N_retrieval, 39) float32 (dense)
      - train_fps.pt          (N_train, 39) float32 (dense)
      - train_indices.pt      (N_train,) long  (idx order aligned with train_fps)
    """
    def __init__(self, subdir: str = "Biosynfoni") -> None:
        self.root = os.path.join(DATASET_ROOT, subdir)
        self._train_fps: Optional[torch.Tensor] = None
        self._train_indices: Optional[torch.Tensor] = None
        self.out_dim = 39

    def setup(self, out_dim=None, max_radius=None):
        # out_dim and max_radius ignored; kept for API compatibility
        # Preload train tensors (tiny memory footprint)
        tr_fp_path = os.path.join(self.root, "train_fps.pt")
        tr_ix_path = os.path.join(self.root, "train_indices.pt")
        if os.path.exists(tr_fp_path) and os.path.exists(tr_ix_path):
            self._train_fps = torch.load(tr_fp_path, weights_only=True)
            self._train_indices = torch.load(tr_ix_path, weights_only=True)
        else:
            self._train_fps, self._train_indices = None, None

    def build_mfp(self, idx: int) -> torch.Tensor:
        """
        Return (39,) tensor for the given dataset idx.
        Requires that train_fps/train_indices were built.
        """
        assert self._train_fps is not None and self._train_indices is not None, \
            "Biosynfoni train_fps.pt / train_indices.pt not found. Run the builder first."
        # find row for idx (indices are sorted; do a binary search or map)
        # For speed, build a map once
        if not hasattr(self, "_idx_to_row"):
            self._idx_to_row = {int(i): r for r, i in enumerate(self._train_indices.tolist())}
        r = self._idx_to_row.get(int(idx))
        if r is None:
            # unseen idx → return zeros (or compute on the fly if you prefer)
            return torch.zeros(self.out_dim, dtype=torch.float32)
        return self._train_fps[r]

    def load_rankingset(self, fp_type) -> torch.Tensor:
        """
        Dense (N, 39) float32 matrix—NO normalization—ready for Tanimoto.
        """
        path = os.path.join(self.root, "rankingset.pt")
        return torch.load(path, weights_only=True)

def make_fp_loader(fp_type: str, entropy_out_dim = 16384):
    if fp_type == "RankingEntropy":
        fp_loader = EntropyFPLoader()
        fp_loader.setup(entropy_out_dim, 6)
        return fp_loader
    elif fp_type == 'Biosynfoni':
        fp_loader = BiosynfoniFPLoader()
        fp_loader.setup()
        return fp_loader
    else:
        return IRFPFPLoader(fp_type)