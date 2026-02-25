# inputs.py
import os
from typing import Iterable, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..core.const import INPUT_TYPES
from .fp_loader import FPLoader
from .arrow_store import ArrowTensorStore


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
