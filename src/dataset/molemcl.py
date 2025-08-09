# ae_datamodule.py

import os
import pickle
from typing import Dict, List, Optional
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage

import pytorch_lightning as pl

# from your util.py (the MoleMCL masking transform)
from ..molemcl.utils import MaskAtom, MaskAtomBalanced

torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])


class AEGraphDataset(Dataset):
    """
    Minimal graph dataset for AE:
      - reads {ae_root}/AE/index.pkl (idx -> dict with keys: 'smiles', 'split', 'duplicate')
      - loads graphs from {ae_root}/AE/Graphs/{idx}.pt (torch_geometric.data.Data)
    Returns ONLY a Data object per sample; masking is applied in the DataModule.
    """
    def __init__(
        self,
        ae_root: str,
        split: str = "train",
        train_on_duplicates: bool = False,
    ):
        ae_dir = os.path.join(ae_root, "AE")
        with open(os.path.join(ae_dir, "index.pkl"), "rb") as f:
            idx_map: Dict[int, dict] = pickle.load(f)

        if not train_on_duplicates:
            idx_map = {i: d for i, d in idx_map.items() if not d.get("duplicate", False)}

        # keep ids for the requested split
        self.ids: List[int] = sorted(i for i, d in idx_map.items() if d.get("split") == split)
        self.graphs_dir = os.path.join(ae_dir, "Graphs")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, ix: int) -> Data:
        idx = self.ids[ix]
        # graphs must be saved with x:int64, edge_attr:int64 and paired edges (i,j),(j,i)
        return torch.load(os.path.join(self.graphs_dir, f"{idx}.pt"), weights_only=True)


class AEDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for MoleMCL-style autoencoder training.

    It wraps AEGraphDataset for train/val/test and applies the MoleMCL mask transform
    (MaskAtom) on-the-fly so batches contain:
      - x, edge_index, edge_attr, batch
      - masked_atom_indices, mask_node_label
      - connected_edge_indices, mask_edge_label  (if mask_edge=True)

    Notes on MaskAtom args:
      - num_atom_type=119 -> writes mask token 119 into x[:,0]; embeddings sized for 120
      - num_edge_type=5   -> writes mask token 5 into edge_attr[:,0]; embeddings sized for 6
    """
    def __init__(
        self,
        ae_root: str,
        batch_size: int = 256,
        num_workers: int = 8,
        persistent_workers: Optional[bool] = None,
        pin_memory: bool = True,
        mask_rate: float = 0.15,
        mask_edge: bool = True,
        train_on_duplicates: bool = False,
        use_balanced_masking: bool = True,
    ):
        super().__init__()
        self.ae_root = ae_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        # default True iff num_workers > 0 unless user overrides
        self.persistent_workers = bool(num_workers > 0) if persistent_workers is None else bool(persistent_workers)
        
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge
        self.train_on_duplicates = train_on_duplicates
        self.use_balanced_masking = use_balanced_masking

        # MoleMCL masking transform (matches model.py constants)
        '''self.mask_transform = MaskAtom(
            num_atom_type=119,  # mask index for atoms
            num_edge_type=5,    # mask index for bonds
            mask_rate=mask_rate,
            mask_edge=mask_edge,
        )'''

        self._train = None
        self._val = None
        self._test = None
        
        self.mask_transform = None
        self.node_class_weights = None
        self.edge_class_weights = None
        self.node_mask_probs = None

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit", "validate") and self._train is None:
            self._train = AEGraphDataset(self.ae_root, split="train", train_on_duplicates=self.train_on_duplicates)
            self._val   = AEGraphDataset(self.ae_root, split="val",   train_on_duplicates=self.train_on_duplicates)
            self.compute_class_weights()
            self._build_mask_transform()
        if stage in (None, "test") and self._test is None:
            self._test  = AEGraphDataset(self.ae_root, split="test",  train_on_duplicates=self.train_on_duplicates)
            if self.mask_transform is None:
                self._build_mask_transform()
    
    def _build_mask_transform(self):
        if self.use_balanced_masking and self.node_mask_probs is not None:
            self.mask_transform = MaskAtomBalanced(
                num_atom_type=119,   # mask id for atoms
                num_edge_type=5,     # mask id for bonds
                mask_rate=self.mask_rate,
                class_probs=self.node_mask_probs,   # tensor (120,)
                mask_edge=self.mask_edge,
            )
        else:
            self.mask_transform = MaskAtom(
                num_atom_type=119,
                num_edge_type=5,
                mask_rate=self.mask_rate,
                mask_edge=self.mask_edge,
            )

    # small wrapper that applies MaskAtom per-sample before collation
    class _MaskedDataset(Dataset):
        def __init__(self, base: Dataset, transform):
            self.base = base
            self.transform = transform
        def __len__(self): return len(self.base)
        def __getitem__(self, i):
            data = self.base[i]
            return self.transform(data)

    def train_dataloader(self):
        ds = self._MaskedDataset(self._train, self.mask_transform)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        ds = self._MaskedDataset(self._val, self.mask_transform)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        ds = self._MaskedDataset(self._test, self.mask_transform)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    @property
    def class_weights(self):
        # convenience accessor
        return getattr(self, "node_class_weights", None), getattr(self, "edge_class_weights", None)

    def compute_class_weights(self):
        """
        Compute inverse-frequency class weights from the TRAIN split:
        - Nodes: atom type = x[:, 0] in [0..119]  (we ignore the mask id 119)
        - Edges: bond type = edge_attr[:, 0] in [0..3] (unique undirected: take [::2])
        Saves:
            self.node_class_weights: (120,) float tensor
            self.edge_class_weights: (4,)   float tensor
        """
        if os.path.exists(os.path.join(self.ae_root, 'ae_cache.pt')):
            data = torch.load(os.path.join(self.ae_root, 'ae_cache.pt'), weights_only=True)
            self.node_class_weights = data['node_weights']   # (120,)
            self.edge_class_weights = data['edge_weights']
            return
        num_atom_type = 120
        num_edge_type = 4  # real bond types only

        node_counts = torch.zeros(num_atom_type, dtype=torch.long)
        edge_counts = torch.zeros(num_edge_type, dtype=torch.long)

        # iterate train dataset only
        for idx in tqdm(self._train.ids, desc='Computing class weights'):
            data = torch.load(os.path.join(self._train.graphs_dir, f"{idx}.pt"), weights_only=True)

            # nodes
            atoms = data.x[:, 0]  # (N,)
            # clamp to the valid range; ignore 119 ([MASK]) if present in any raw graphs
            atoms = atoms.clamp(min=0, max=num_atom_type - 1)
            node_counts.index_add_(0, atoms, torch.ones_like(atoms, dtype=torch.long))

            # edges (unique undirected: [::2]); guard empty graphs
            if data.edge_attr.numel() > 0:
                e_types = data.edge_attr[::2, 0]  # (E_unique,)
                e_types = e_types.clamp(min=0, max=num_edge_type - 1)
                edge_counts.index_add_(0, e_types, torch.ones_like(e_types, dtype=torch.long))

        # avoid zero division
        node_counts = node_counts + 1
        edge_counts = edge_counts + 1

        node_weights = 1.0 / node_counts.float()
        edge_weights = 1.0 / edge_counts.float()

        # (optional) normalize to have mean ~1
        node_weights = node_weights * (node_weights.numel() / node_weights.sum())
        edge_weights = edge_weights * (edge_weights.numel() / edge_weights.sum())

        self.node_class_weights = node_weights   # (120,)
        self.edge_class_weights = edge_weights   # (4,)
        
        probs = (1.0 / node_counts.float())
        probs[119] = 0.0                       # never sample the mask token as a target
        probs = probs / probs.sum().clamp_min(1e-12)
        self.node_mask_probs = probs 

        cache_path = os.path.join(self.ae_root, 'ae_cache.pt')
        torch.save({
            'node_weights': node_weights,
            'edge_weights': edge_weights,
            'node_mask_probs': probs
            }, cache_path
        )