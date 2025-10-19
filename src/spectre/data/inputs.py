import os
import io
from typing import Iterable, Dict, Optional

import torch
import torch.nn.functional as F

import lmdb

from ..core.const import INPUT_TYPES
from .fp_loader import FPLoader


class _LMDBModalityStore:
    """
    Open the LMDB env lazily per-process (PID-aware) to avoid using a closed/
    invalid handle after DataLoader forks/spawns workers.
    """
    def __init__(self, path: str):
        if lmdb is None:
            raise RuntimeError("lmdb is not installed. `pip install lmdb`")
        if not os.path.isdir(path):
            raise FileNotFoundError(f"LMDB path not found: {path}")
        self.path = path
        self._env = None
        self._pid = None

    def _get_env(self):
        pid = os.getpid()
        if self._env is None or self._pid != pid:
            # (Re)open for this process
            self._env = lmdb.open(
                self.path,
                readonly=True,
                lock=False,        # many readers, no writer
                readahead=True,
                max_readers=4096,
                subdir=True
            )
            self._pid = pid
        return self._env

    def get_tensor(self, idx: int, weights_only: bool = True) -> torch.Tensor:
        key = str(idx).encode("utf-8")
        env = self._get_env()

        # One-shot read transaction
        try:
            with env.begin(write=False) as txn:
                buf = txn.get(key)
        except lmdb.Error:
            # If env became invalid (e.g., after fork), reopen and retry once
            self._env = None
            env = self._get_env()
            with env.begin(write=False) as txn:
                buf = txn.get(key)

        if buf is None:
            raise KeyError(f"Key {idx} not found in {self.path}")
        bio = io.BytesIO(buf)
        return torch.load(bio, map_location="cpu")


class SpectralInputLoader:
    '''
    Represents the SPECTRE input data types.

    - HSQC NMR ('hsqc')
    - H NMR ('h_nmr')
    - C NMR ('c_nmr')
    - MS/MS ('mass_spec')
    - Molecular Weight ('mw')
    '''
    def __init__(self, root: str, data_dict: dict, split: Optional[str] = None, dtype=torch.float32):
        '''
        In index.pkl, it is stored idx: data_dict pairs. Feed this in for initialization.
        If LMDB shards exist under {root}/_lmdb/{split}/{MODALITY}.lmdb, we will read from them.
        Fallback to filesystem .pt files otherwise.
        '''
        self.root = root
        self.data_dict = data_dict
        self.dtype = dtype
        self.kwargs = {'weights_only': True}

        self.split = split  # 'train'|'val'|'test' (optional for LMDB discovery)

        # Auto-detect LMDB layout: {root}/_lmdb/{split}/{mod}.lmdb
        self._lmdb = {}
        lmdb_base = os.path.join(self.root, "_lmdb")
        if self.split is not None and os.path.isdir(lmdb_base):
            lmdb_split_dir = os.path.join(lmdb_base, self.split)
            if os.path.isdir(lmdb_split_dir):
                for mod in ("HSQC_NMR", "H_NMR", "C_NMR", "MassSpec"):
                    path = os.path.join(lmdb_split_dir, f"{mod}.lmdb")
                    if os.path.isdir(path):
                        self._lmdb[mod] = _LMDBModalityStore(path)

    # ---- public API ----
    def load(self, idx, input_types: Iterable[INPUT_TYPES], jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        '''
        Load spectral inputs.
        Returns dict of requested input types and their data.
        '''
        data_inputs = {}
        for input_type in input_types:
            data_inputs.update(getattr(self, f'_load_{input_type}')(idx, jittering))
        return data_inputs

    # ---- helpers ----
    def _get_tensor(self, idx: int, modality_dir: str, filename: str) -> torch.Tensor:
        """
        Read a tensor either from LMDB shard (fast path) or from filesystem .pt (fallback).
        `modality_dir` is e.g. 'HSQC_NMR', 'H_NMR', 'C_NMR', 'MassSpec'.
        """
        if modality_dir in self._lmdb:
            t = self._lmdb[modality_dir].get_tensor(idx)
        else:
            t = torch.load(os.path.join(self.root, modality_dir, filename), **self.kwargs)
        return t.to(dtype=self.dtype)

    # ---- individual modality loaders ----
    def _load_hsqc(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        hsqc = self._get_tensor(idx, 'HSQC_NMR', f'{idx}.pt')
        if jittering > 0:
            hsqc[:,0] = hsqc[:,0] + torch.randn_like(hsqc[:,0]) * jittering
            hsqc[:,1] = hsqc[:,1] + torch.randn_like(hsqc[:,1]) * jittering * 0.1
        return {'hsqc': hsqc}

    def _load_c_nmr(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        c_nmr = self._get_tensor(idx, 'C_NMR', f'{idx}.pt')
        c_nmr = c_nmr.view(-1,1)                   # (N,1)
        c_nmr = F.pad(c_nmr, (0,2), "constant", 0) # -> (N,3)
        if jittering > 0:
            c_nmr = c_nmr + torch.randn_like(c_nmr) * jittering
        return {'c_nmr': c_nmr}

    def _load_h_nmr(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        h_nmr = self._get_tensor(idx, 'H_NMR', f'{idx}.pt')
        h_nmr = h_nmr.view(-1,1)                    # (N,1)
        h_nmr = F.pad(h_nmr, (1,1), "constant", 0)  # -> (N,3)
        if jittering > 0:
            h_nmr = h_nmr + torch.randn_like(h_nmr) * jittering * 0.1
        return {'h_nmr': h_nmr}

    def _load_mass_spec(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        mass_spec = self._get_tensor(idx, 'MassSpec', f'{idx}.pt')
        mass_spec = F.pad(mass_spec, (0,1), "constant", 0)
        if jittering > 0:
            noise = torch.zeros_like(mass_spec)
            noise[:, 0] = torch.randn_like(mass_spec[:, 0]) * mass_spec[:, 0] / 100_000  # jitter m/z
            noise[:, 1] = torch.randn_like(mass_spec[:, 1]) * mass_spec[:, 1] / 10
            mass_spec = mass_spec + noise
        return {'mass_spec': mass_spec}

    def _load_mw(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        return {'mw': torch.tensor(self.data_dict[idx]['mw'], dtype=self.dtype)}


class MFInputLoader:
    '''
    The Morgan Fingerprint groundtruth loader.
    '''
    def __init__(self, fp_loader: FPLoader):
        self.fp_loader = fp_loader

    def load(self, idx: int) -> torch.Tensor:
        return self.fp_loader.build_mfp(idx)
