import os
import pickle
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

DTYPE_MAP = {
    "float32": np.float32,
    "float64": np.float64,
    "int64": np.int64,
    "int32": np.int32,
    "int16": np.int16,
    "uint8": np.uint8,
}


def decode_array(buf) -> np.ndarray:
    mv = memoryview(buf)
    b = mv.tobytes()
    first = b.index(b"|")
    second = b.index(b"|", first + 1)
    third = b.index(b"|", second + 1)
    dtype_str = b[:first].decode("ascii")
    ndim = int(b[first + 1 : second].decode("ascii"))
    shape_str = b[second + 1 : third].decode("ascii")
    shape = (
        tuple(int(x) for x in shape_str.split(",")) if ndim > 0 and shape_str else ()
    )
    dt = DTYPE_MAP[dtype_str]
    arr = np.frombuffer(mv[third + 1 :], dtype=dt)
    if shape:
        arr = arr.reshape(shape)
    return arr


class _PIDLMDB:
    def __init__(self, path: str):
        self.path = path
        self._env = None
        self._pid = None

    def _get_env(self):
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

    def get(self, idx: int):
        env = self._get_env()
        key = str(idx).encode("utf-8")
        with env.begin(write=False, buffers=True) as txn:
            v = txn.get(key)
        return v  # may be None

    def has(self, idx: int) -> bool:
        return self.get(idx) is not None


class HSQCSelfiesDataset(Dataset):
    def __init__(
        self,
        dataset_root="/data",
        split="train",
        normalize_intensity=True,
        require_hsqc=True,
    ):
        self.dataset_root = dataset_root
        self.split = split
        self.normalize_intensity = normalize_intensity
        self.require_hsqc = require_hsqc

        self.hsqc_lmdb = _PIDLMDB(
            os.path.join(dataset_root, "_lmdb", split, "HSQC_NMR.lmdb")
        )
        self.sf_lmdb = _PIDLMDB(
            os.path.join(dataset_root, "_lmdb", split, "SELFIES_IDs.lmdb")
        )

        # Build explicit id list from index.pkl
        index_path = os.path.join(dataset_root, "index.pkl")
        with open(index_path, "rb") as f:
            index = pickle.load(f)

        ids = []
        missing_hsqc = 0
        missing_selfies = 0
        skipped_no_hsqc_flag = 0

        for idx, entry in index.items():
            if entry.get("split") != split:
                continue
            if require_hsqc and not entry.get("has_hsqc", False):
                skipped_no_hsqc_flag += 1
                continue

            i = int(idx)

            # Ensure both LMDBs actually contain the key
            if self.hsqc_lmdb.get(i) is None:
                missing_hsqc += 1
                continue
            if self.sf_lmdb.get(i) is None:
                missing_selfies += 1
                continue

            ids.append(i)

        self.ids = ids
        print(
            f"[{split}] dataset size={len(self.ids)} "
            f"(skipped_no_hsqc_flag={skipped_no_hsqc_flag}, "
            f"missing_hsqc={missing_hsqc}, missing_selfies={missing_selfies})"
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]

        hsqc_buf = self.hsqc_lmdb.get(idx)
        sf_buf = self.sf_lmdb.get(idx)
        if hsqc_buf is None or sf_buf is None:
            raise KeyError(f"Missing LMDB key idx={idx}")

        hsqc = decode_array(hsqc_buf).astype(np.float32, copy=False)  # (N,3)
        ids = decode_array(sf_buf).astype(np.int64, copy=False)  # (T,)

        hsqc = torch.from_numpy(np.array(hsqc, copy=True))
        ids = torch.from_numpy(np.array(ids, copy=True))

        if self.normalize_intensity:
            I = hsqc[:, 2]
            hsqc[:, 2] = I / (I.abs().max() + 1e-6)

        return hsqc, ids
