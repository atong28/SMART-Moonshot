# inputs.py
import os
from typing import Iterable, Dict, Optional, Any
import numpy as np
import pickle
import torch
import torch.nn.functional as F

from ..core.const import INPUT_TYPES, ATOM_TYPES, BOND_TYPES
from .fp_loader import FPLoader
from .arrow_store import ArrowTensorStore
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

class SpectralInputLoader:
    '''
    Represents the MARINA input data types.

    - HSQC NMR ('hsqc')
    - H NMR ('h_nmr')
    - C NMR ('c_nmr')
    - MS/MS ('mass_spec')
    - Molecular Weight ('mw')
    '''
    def __init__(self, root: str, data_dict: dict, split: Optional[str] = None, dtype=torch.float32):
        '''
        In index.pkl, it is stored idx: data_dict pairs. Feed this in for initialization.
        We read from Arrow shards under {root}/arrow/{split}/{MODALITY}.parquet.
        '''
        self.root = root
        self.data_dict = data_dict
        self.dtype = dtype

        self.split = split  # 'train'|'val'|'test'

        # Arrow layout: {root}/arrow/{split}/{mod}.parquet
        self._arrow = {}
        arrow_base = os.path.join(self.root, "arrow")
        if self.split is None:
            raise ValueError("SpectralInputLoader requires split for Arrow discovery.")
        arrow_split_dir = os.path.join(arrow_base, self.split)
        if not os.path.isdir(arrow_split_dir):
            raise FileNotFoundError(f"Arrow split directory not found: {arrow_split_dir}")
        for mod in ("HSQC_NMR", "H_NMR", "C_NMR", "MassSpec"):
            path = os.path.join(arrow_split_dir, f"{mod}.parquet")
            if os.path.isfile(path):
                self._arrow[mod] = ArrowTensorStore(path)

    # ---- public API ----
    def load(self, idx, input_types: Iterable[INPUT_TYPES], jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        '''
        Load spectral inputs from Arrow shards.
        Returns dict of requested input types and their data.
        '''
        data_inputs = {}
        for input_type in input_types:
            data_inputs.update(getattr(self, f'_load_{input_type}')(idx, jittering))
        return data_inputs

    # ---- helpers ----
    def _get_tensor(self, idx: int, modality_dir: str) -> torch.Tensor:
        """
        Read a tensor from Arrow shard (required).
        `modality_dir` is e.g. 'HSQC_NMR', 'H_NMR', 'C_NMR', 'MassSpec'.
        """
        if modality_dir not in self._arrow:
            raise FileNotFoundError(f"Missing Arrow shard for modality {modality_dir}")
        t = self._arrow[modality_dir].get_tensor(idx)
        return t.to(dtype=self.dtype)

    # ---- individual modality loaders ----
    def _load_hsqc(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        hsqc = self._get_tensor(idx, 'HSQC_NMR')
        if jittering > 0:
            hsqc[:,0] = hsqc[:,0] + torch.randn_like(hsqc[:,0]) * jittering
            hsqc[:,1] = hsqc[:,1] + torch.randn_like(hsqc[:,1]) * jittering * 0.1
        return {'hsqc': hsqc}

    def _load_c_nmr(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def _load_h_nmr(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def _load_mass_spec(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def _load_mw(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

class MARINAInputLoader(SpectralInputLoader):
    def _load_mw(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        mw = torch.tensor(self.data_dict[idx]['mw'], dtype=self.dtype)
        mw = mw.view(1, 1)
        return {'mw': mw}

    def _load_mass_spec(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        mass_spec = self._get_tensor(idx, 'MassSpec')
        if jittering > 0:
            noise = torch.zeros_like(mass_spec)
            noise[:, 0].copy_(torch.randn_like(mass_spec[:, 0]) * mass_spec[:, 0] / 100_000)
            noise[:, 1].copy_(torch.randn_like(mass_spec[:, 1]) * mass_spec[:, 1] / 10)
            mass_spec = mass_spec + noise
        return {'mass_spec': mass_spec}

    def _load_c_nmr(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        c_nmr = self._get_tensor(idx, 'C_NMR')
        c_nmr = c_nmr.view(-1,1)                   # (N,1)
        if jittering > 0:
            c_nmr = c_nmr + torch.randn_like(c_nmr) * jittering
        return {'c_nmr': c_nmr}

    def _load_h_nmr(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        h_nmr = self._get_tensor(idx, 'H_NMR')
        h_nmr = h_nmr.view(-1,1)                    # (N,1)
        if jittering > 0:
            h_nmr = h_nmr + torch.randn_like(h_nmr) * jittering * 0.1
        return {'h_nmr': h_nmr}

class SPECTREInputLoader(SpectralInputLoader):
    def _load_mw(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        return {'mw': torch.tensor(self.data_dict[idx]['mw'], dtype=self.dtype)}

    def _load_mass_spec(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        mass_spec = self._get_tensor(idx, 'MassSpec')
        mass_spec = F.pad(mass_spec, (0,1), "constant", 0)
        if jittering > 0:
            noise = torch.zeros_like(mass_spec)
            noise[:, 0].copy_(torch.randn_like(mass_spec[:, 0]) * mass_spec[:, 0] / 100_000)
            noise[:, 1].copy_(torch.randn_like(mass_spec[:, 1]) * mass_spec[:, 1] / 10)
            mass_spec = mass_spec + noise
        return {'mass_spec': mass_spec}

    def _load_c_nmr(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        c_nmr = self._get_tensor(idx, 'C_NMR')
        c_nmr = c_nmr.view(-1,1)                   # (N,1)
        c_nmr = F.pad(c_nmr, (0,2), "constant", 0) # -> (N,3)
        if jittering > 0:
            c_nmr = c_nmr + torch.randn_like(c_nmr) * jittering
        return {'c_nmr': c_nmr}

    def _load_h_nmr(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        h_nmr = self._get_tensor(idx, 'H_NMR')
        h_nmr = h_nmr.view(-1,1)                    # (N,1)
        h_nmr = F.pad(h_nmr, (1,1), "constant", 0)  # -> (N,3)
        if jittering > 0:
            h_nmr = h_nmr + torch.randn_like(h_nmr) * jittering * 0.1
        return {'h_nmr': h_nmr}
    
class MFInputLoader:
    '''
    The Morgan Fingerprint groundtruth loader.
    '''
    def __init__(self, fp_loader: FPLoader):
        self.fp_loader = fp_loader

    def load(self, idx: int) -> torch.Tensor:
        return self.fp_loader.build_mfp(idx)

class GraphInputLoader:
    def __init__(
        self,
        data_dict: dict[int, Any],
        root: Optional[str] = None,
        split: Optional[str] = None,
        use_arrow: Optional[bool] = None,
    ):
        """
        If use_arrow is:
          - True:  load from Arrow shards (requires root & split)
          - False: compute on-the-fly from SMILES
          - None:  auto-detect Arrow shards under <root>/arrow/<split>/ and use
                   them if present, otherwise fall back to SMILES.
        """
        self.data_dict = data_dict
        self.root = root
        self.split = split
        self._use_arrow = False
        self._arrow_x: Optional[ArrowTensorStore] = None
        self._arrow_edge_index: Optional[ArrowTensorStore] = None
        self._arrow_edge_attr: Optional[ArrowTensorStore] = None

        if use_arrow is True:
            if self.root is None or self.split is None:
                raise ValueError("GraphInputLoader(use_arrow=True) requires root and split.")
            if not self._init_arrow_stores():
                raise FileNotFoundError(
                    f"Graph Arrow shards not found under {os.path.join(self.root, 'arrow', self.split)}"
                )
            self._use_arrow = True
        elif use_arrow is None and self.root is not None and self.split is not None:
            if self._init_arrow_stores():
                self._use_arrow = True

    def _init_arrow_stores(self) -> bool:
        arrow_base = os.path.join(self.root, "arrow")
        arrow_split_dir = os.path.join(arrow_base, self.split)
        x_path = os.path.join(arrow_split_dir, "GraphX.parquet")
        ei_path = os.path.join(arrow_split_dir, "GraphEdgeIndex.parquet")
        ea_path = os.path.join(arrow_split_dir, "GraphEdgeAttr.parquet")
        if not (os.path.isfile(x_path) and os.path.isfile(ei_path) and os.path.isfile(ea_path)):
            return False
        self._arrow_x = ArrowTensorStore(x_path)
        self._arrow_edge_index = ArrowTensorStore(ei_path)
        self._arrow_edge_attr = ArrowTensorStore(ea_path)
        return True

    def _load_from_smiles(self, idx: int) -> Data:
        smiles = self.data_dict[idx]['smiles']
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        N = mol.GetNumAtoms()
        nodes = [ATOM_TYPES[atom.GetSymbol()] for atom in mol.GetAtoms()]
        row, col, edge_type = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_type += 2 * [BOND_TYPES[bond.GetBondType()] + 1]
        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        edge_attr = F.one_hot(edge_type, num_classes=len(BOND_TYPES) + 1).to(torch.float)
        perm = (edge_index[0] * N + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_attr = edge_attr[perm]
        x = F.one_hot(torch.tensor(nodes), num_classes=len(ATOM_TYPES)).float()
        return x, edge_index, edge_attr, smiles

    def _load_from_arrow(self, idx: int) -> Data:
        if self._arrow_x is None or self._arrow_edge_index is None or self._arrow_edge_attr is None:
            raise RuntimeError("Arrow stores not initialized for GraphInputLoader.")
        x = self._arrow_x.get_tensor(idx)
        edge_index = self._arrow_edge_index.get_tensor(idx).to(dtype=torch.long)
        edge_attr = self._arrow_edge_attr.get_tensor(idx)
        smiles = self.data_dict[idx]['smiles']
        return x, edge_index, edge_attr, smiles

    def load(self, idx: int) -> Data:
        if self._use_arrow:
            return self._load_from_arrow(idx)
        return self._load_from_smiles(idx)

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

class RDKitMFInputLoader:
    def __init__(self, data_dict: dict[int, Any]):
        self.data_dict = data_dict

    def load(self, idx: int) -> torch.Tensor:
        smiles = self.data_dict[idx]['smiles']
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: raise ValueError(f"Invalid SMILES: {smiles}")
        fp = gen.GetFingerprint(mol)
        y = torch.tensor(np.asarray(fp, dtype=np.int8)).unsqueeze(0)
        return y