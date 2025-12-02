# inputs.py
import os
from typing import Iterable, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import lmdb

from ..core.const import INPUT_TYPES
from .fp_loader import FPLoader


class _LMDBModalityStore:
    """
    Open the LMDB env lazily per-process (PID-aware) to avoid using a closed/
    invalid handle after DataLoader forks/spawns workers.

    Values are raw bytes with header:
      b"{dtype}|{ndim}|{d0},{d1},...|<raw-bytes>"
    """
    def __init__(self, path: str):
        if not os.path.isdir(path):
            raise FileNotFoundError(f"LMDB path not found: {path}")
        self.path = path
        self._env = None
        self._pid = None

    def _get_env(self):
        pid = os.getpid()
        if self._env is None or self._pid != pid:
            self._env = lmdb.open(
                self.path,
                readonly=True,
                lock=False,        # many readers, no writer
                readahead=False,
                max_readers=4096,
                subdir=True
            )
            self._pid = pid
        return self._env

    def get_tensor(self, idx: int) -> torch.Tensor:
        key = str(idx).encode("utf-8")
        env = self._get_env()

        # Use buffers=True to get a memoryview; we still copy a tiny header slice
        with env.begin(write=False, buffers=True) as txn:
            buf = txn.get(key)
        if buf is None:
            raise KeyError(f"Key {idx} not found in {self.path}")

        mv = memoryview(buf)
        b = mv.tobytes()  # small header copy
        first = b.index(b'|')
        second = b.index(b'|', first + 1)
        third = b.index(b'|', second + 1)
        dtype_str = b[:first].decode('ascii')
        ndim = int(b[first+1:second].decode('ascii'))
        shape = tuple(int(x) for x in b[second+1:third].decode('ascii').split(',')) if ndim > 0 else ()
        _DT = {
            'float32': np.float32, 'float64': np.float64,
            'int64': np.int64, 'int32': np.int32, 'int16': np.int16,
            'uint8': np.uint8
        }
        dt = _DT[dtype_str]
        arr = np.frombuffer(mv[third+1:], dtype=dt, count=(int(np.prod(shape)) if shape else -1))
        if shape:
            arr = arr.reshape(shape)
        return torch.from_numpy(arr.copy())  # zero-copy view


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
        We read from LMDB shards under {root}/_lmdb/{split}/{MODALITY}.lmdb.
        '''
        self.root = root
        self.data_dict = data_dict
        self.dtype = dtype

        self.split = split  # 'train'|'val'|'test'

        # Auto-detect LMDB layout: {root}/_lmdb/{split}/{mod}.lmdb
        self._lmdb = {}
        lmdb_base = os.path.join(self.root, "_lmdb")
        if self.split is None:
            raise ValueError("SpectralInputLoader requires split for LMDB discovery.")
        lmdb_split_dir = os.path.join(lmdb_base, self.split)
        if not os.path.isdir(lmdb_split_dir):
            raise FileNotFoundError(f"LMDB split directory not found: {lmdb_split_dir}")
        for mod in ("HSQC_NMR", "H_NMR", "C_NMR", "MassSpec"):
            path = os.path.join(lmdb_split_dir, f"{mod}.lmdb")
            if os.path.isdir(path):
                self._lmdb[mod] = _LMDBModalityStore(path)

    # ---- public API ----
    def load(self, idx, input_types: Iterable[INPUT_TYPES], jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        '''
        Load spectral inputs (LMDB-only).
        Returns dict of requested input types and their data.
        '''
        data_inputs = {}
        for input_type in input_types:
            data_inputs.update(getattr(self, f'_load_{input_type}')(idx, jittering))
        return data_inputs

    # ---- helpers ----
    def _get_tensor(self, idx: int, modality_dir: str) -> torch.Tensor:
        """
        Read a tensor from LMDB shard (required).
        `modality_dir` is e.g. 'HSQC_NMR', 'H_NMR', 'C_NMR', 'MassSpec'.
        """
        if modality_dir not in self._lmdb:
            raise FileNotFoundError(f"Missing LMDB shard for modality {modality_dir}")
        t = self._lmdb[modality_dir].get_tensor(idx)
        return t.to(dtype=self.dtype)

    # ---- individual modality loaders ----
    def _load_hsqc(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        hsqc = self._get_tensor(idx, 'HSQC_NMR')
        if jittering > 0:
            hsqc[:,0] = hsqc[:,0] + torch.randn_like(hsqc[:,0]) * jittering
            hsqc[:,1] = hsqc[:,1] + torch.randn_like(hsqc[:,1]) * jittering * 0.1
        return {'hsqc': hsqc}

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

    def _load_mass_spec(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        mass_spec = self._get_tensor(idx, 'MassSpec')
        mass_spec = F.pad(mass_spec, (0,1), "constant", 0)
        if jittering > 0:
            noise = torch.zeros_like(mass_spec)
            noise[:, 0].copy_(torch.randn_like(mass_spec[:, 0]) * mass_spec[:, 0] / 100_000)
            noise[:, 1].copy_(torch.randn_like(mass_spec[:, 1]) * mass_spec[:, 1] / 10)
            mass_spec = mass_spec + noise
        return {'mass_spec': mass_spec}

    def _load_mw(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        return {'mw': torch.tensor(self.data_dict[idx]['mw'], dtype=self.dtype)}

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

class MFInputLoader:
    '''
    The Morgan Fingerprint groundtruth loader.
    '''
    def __init__(self, fp_loader: FPLoader):
        self.fp_loader = fp_loader

    def load(self, idx: int) -> torch.Tensor:
        return self.fp_loader.build_mfp(idx)
