import pickle
import numpy as np
import torch
import os
import time

from ..core.const import DATASET_ROOT, CODE_ROOT
from .fp_utils import compute_entropy, count_circular_substructures

class FPLoader:
    def __init__(self) -> None:
        raise NotImplementedError()
    
    def setup(self, out_dim, max_radius):
        raise NotImplementedError() 
    
    def build_mfp(self, idx: int) -> torch.Tensor:
        raise NotImplementedError()
    
    def load_rankingset(self):
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
    
    def load_rankingset(self):
        rankingset_path = os.path.join(DATASET_ROOT, 'rankingset.pt')
        return torch.load(rankingset_path, weights_only=True)