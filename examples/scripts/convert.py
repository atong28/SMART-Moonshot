#!/usr/bin/env python
"""
Convert between LMDB-based dataset format and regular JSONL dataset format.

This script supports bidirectional conversion:
  - from_lmdb: Converts LMDB shards to JSONL files
  - to_lmdb: Converts JSONL files to LMDB shards

Regular files (index.pkl, retrieval.pkl, metadata.json, etc.) are copied as-is.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import lmdb
import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPLITS = ("train", "val", "test")
# lmdb dirs to convert
MOD_DIRS = {
    "hsqc": "HSQC_NMR",
    "h_nmr": "H_NMR",
    "c_nmr": "C_NMR",
    "mass_spec": "MassSpec",
    "hmbc": "HMBC_NMR",
    "cosy": "COSY_NMR",
    "fragidx": "FragIdx",
}

# files/dirs to copy as is
REGULAR_FILES = [
    "index.pkl",
    "retrieval.pkl",
    "metadata.json",
    "count_hashes_under_radius_6.pkl",
]
REGULAR_DIRS = [
    "RankingEntropy",
]


# ---------------------------------------------------------------------------
# Tensor encoding/decoding (matching existing format)
# ---------------------------------------------------------------------------

def encode_tensor_raw(t: torch.Tensor) -> bytes:
    """Encode tensor to bytes with header: dtype|ndim|shape|<raw-bytes>"""
    t = t.contiguous()
    arr = np.ascontiguousarray(t.detach().cpu().numpy())
    header = f"{arr.dtype.name}|{arr.ndim}|{','.join(map(str, arr.shape))}|".encode("ascii")
    return header + memoryview(arr)


def decode_tensor_from_bytes(buf: bytes) -> torch.Tensor:
    """Decode tensor from bytes with header format"""
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
    return torch.from_numpy(arr.copy())


# ---------------------------------------------------------------------------
# Tensor <-> List conversions
# ---------------------------------------------------------------------------

def tensor_to_list(tensor: torch.Tensor) -> List[Any]:
    """Convert tensor to Python list (handles 1D and 2D tensors)"""
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 1:
        return arr.tolist()
    elif arr.ndim == 2:
        return arr.tolist()
    else:
        raise ValueError(f"Unsupported tensor shape: {arr.shape}")


def list_to_tensor(data: List[Any], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert Python list to tensor"""
    if not data:
        return None
    arr = np.array(data, dtype=dtype.numpy_dtype if hasattr(dtype, 'numpy_dtype') else np.float32)
    return torch.from_numpy(arr)


# ---------------------------------------------------------------------------
# LMDB -> JSONL conversion
# ---------------------------------------------------------------------------

class LMDBReader:
    """Read-only LMDB environment that can be reused"""
    
    def __init__(self, lmdb_path: str):
        self.lmdb_path = lmdb_path
        self._env = None
    
    def _get_env(self):
        if self._env is None:
            if not os.path.isdir(self.lmdb_path):
                return None
            self._env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                max_readers=4096,
                subdir=True
            )
        return self._env
    
    def get_tensor(self, idx: int) -> Optional[torch.Tensor]:
        """Read a tensor from LMDB by index"""
        env = self._get_env()
        if env is None:
            return None
        
        key = str(idx).encode("utf-8")
        try:
            with env.begin(write=False, buffers=True) as txn:
                buf = txn.get(key)
            if buf is None:
                return None
            return decode_tensor_from_bytes(bytes(buf))
        except Exception as e:
            print(f"Warning: Error reading idx {idx} from {self.lmdb_path}: {e}")
            return None
    
    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None


def convert_from_lmdb(input_dir: str, output_dir: str):
    """Convert LMDB dataset to JSONL format"""
    print(f"Converting from LMDB format: {input_dir} -> {output_dir}")
    
    # Load index
    index_path = os.path.join(input_dir, "index.pkl")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"index.pkl not found in {input_dir}")
    
    with open(index_path, "rb") as f:
        index = pickle.load(f)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy regular files
    print("Copying regular files...")
    for filename in REGULAR_FILES:
        src = os.path.join(input_dir, filename)
        if os.path.exists(src):
            dst = os.path.join(output_dir, filename)
            shutil.copy2(src, dst)
            print(f"  Copied {filename}")
    
    # Copy regular directories
    for dirname in REGULAR_DIRS:
        src = os.path.join(input_dir, dirname)
        if os.path.exists(src):
            dst = os.path.join(output_dir, dirname)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"  Copied {dirname}/")
    
    # Convert LMDB shards to JSONL
    lmdb_base = os.path.join(input_dir, "_lmdb")
    if not os.path.exists(lmdb_base):
        print(f"Warning: _lmdb directory not found in {input_dir}, skipping LMDB conversion")
        return
    
    # Group indices by split
    split_indices: Dict[str, List[int]] = {split: [] for split in SPLITS}
    for idx_str, entry in index.items():
        idx = int(idx_str)
        split = entry.get("split", "train")
        if split in split_indices:
            split_indices[split].append(idx)
    
    # Process each split
    for split in SPLITS:
        indices = sorted(split_indices[split])
        if not indices:
            print(f"Skipping {split} (no entries)")
            continue
        
        jsonl_path = os.path.join(output_dir, f"{split}.jsonl")
        print(f"\nConverting {split} ({len(indices)} entries) -> {jsonl_path}")
        
        # Open LMDB readers for this split (reuse across all indices)
        split_lmdb_dir = os.path.join(lmdb_base, split)
        readers: Dict[str, LMDBReader] = {}
        for mod_key, mod_dir in MOD_DIRS.items():
            lmdb_path = os.path.join(split_lmdb_dir, f"{mod_dir}.lmdb")
            readers[mod_key] = LMDBReader(lmdb_path)
        
        try:
            with open(jsonl_path, "w") as f:
                for idx in tqdm(indices, desc=f"{split}"):
                    entry = index[idx]
                    
                    # Build JSON entry
                    json_entry = {
                        "idx": idx,
                        "smiles": entry.get("smiles", ""),
                        "split": split,
                    }
                    
                    # Copy all metadata from index entry
                    for key, value in entry.items():
                        if key not in ["smiles", "split"]:
                            json_entry[key] = value
                    
                    # Load tensors from LMDB and convert to lists
                    for mod_key, mod_dir in MOD_DIRS.items():
                        reader = readers[mod_key]
                        tensor = reader.get_tensor(idx)
                        if tensor is not None:
                            if mod_key == "fragidx":
                                # FragIdx is int32 array
                                json_entry["fragidx"] = tensor.detach().cpu().numpy().astype(int).tolist()
                            elif mod_key in ["h_nmr", "c_nmr"]:
                                # 1D tensors (or 2D with shape (N, 1)) -> list of floats
                                if tensor.ndim == 2 and tensor.shape[1] == 1:
                                    tensor = tensor.squeeze(1)
                                json_entry[mod_key] = tensor_to_list(tensor)
                            elif tensor.ndim == 2 and tensor.shape[1] == 2:
                                # 2D with 2 columns: hsqc, cosy, hmbc, mass_spec -> list of [x, y] pairs
                                json_entry[mod_key] = tensor_to_list(tensor)
                            else:
                                # Fallback: convert to list
                                json_entry[mod_key] = tensor_to_list(tensor)
                    
                    f.write(json.dumps(json_entry) + "\n")
        finally:
            # Close all readers
            for reader in readers.values():
                reader.close()
        
        print(f"  Wrote {len(indices)} entries to {jsonl_path}")


# ---------------------------------------------------------------------------
# JSONL -> LMDB conversion
# ---------------------------------------------------------------------------

@dataclass
class LMDBWriter:
    """Append tensors to an existing LMDB shard, auto-resizing as needed."""
    
    path: str
    batch_size: int = 1024
    initial_map_size: int = 1 << 40  # ~1 TB
    
    def __post_init__(self):
        os.makedirs(self.path, exist_ok=True)
        self.env = lmdb.open(
            self.path,
            map_size=self.initial_map_size,
            subdir=True,
            lock=True,
            writemap=False,
            map_async=False,
            max_dbs=1,
        )
        self._batch: List[Tuple[int, bytes]] = []
    
    def add(self, idx: int, tensor: torch.Tensor):
        if tensor is None:
            return
        self._batch.append((idx, encode_tensor_raw(tensor)))
        if len(self._batch) >= self.batch_size:
            self._flush()
    
    def _flush(self):
        if not self._batch:
            return
        while True:
            try:
                with self.env.begin(write=True) as txn:
                    for idx, buf in self._batch:
                        txn.put(str(idx).encode("utf-8"), buf)
                break
            except lmdb.MapFullError:
                current = self.env.info()["map_size"]
                self.env.set_mapsize(int(current * 1.5))
        self._batch.clear()
    
    def close(self):
        self._flush()
        self.env.sync()
        self.env.close()


def convert_to_lmdb(input_dir: str, output_dir: str):
    """Convert JSONL dataset to LMDB format"""
    print(f"Converting to LMDB format: {input_dir} -> {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy regular files
    print("Copying regular files...")
    for filename in REGULAR_FILES:
        src = os.path.join(input_dir, filename)
        if os.path.exists(src):
            dst = os.path.join(output_dir, filename)
            shutil.copy2(src, dst)
            print(f"  Copied {filename}")
    
    # Copy regular directories
    for dirname in REGULAR_DIRS:
        src = os.path.join(input_dir, dirname)
        if os.path.exists(src):
            dst = os.path.join(output_dir, dirname)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"  Copied {dirname}/")
    
    # Load index if it exists (for validation)
    index_path = os.path.join(input_dir, "index.pkl")
    index = {}
    if os.path.exists(index_path):
        with open(index_path, "rb") as f:
            index = pickle.load(f)
    
    # Create LMDB writers for each split and modality
    lmdb_base = os.path.join(output_dir, "_lmdb")
    os.makedirs(lmdb_base, exist_ok=True)
    
    split_writers: Dict[str, Dict[str, LMDBWriter]] = {}
    
    # Process each split
    for split in SPLITS:
        jsonl_path = os.path.join(input_dir, f"{split}.jsonl")
        if not os.path.exists(jsonl_path):
            print(f"Skipping {split} (no {split}.jsonl file)")
            continue
        
        print(f"\nConverting {split} from {jsonl_path}")
        
        # Initialize writers for this split
        split_dir = os.path.join(lmdb_base, split)
        os.makedirs(split_dir, exist_ok=True)
        split_writers[split] = {}
        for mod_key, mod_dir in MOD_DIRS.items():
            lmdb_path = os.path.join(split_dir, f"{mod_dir}.lmdb")
            split_writers[split][mod_key] = LMDBWriter(lmdb_path)
        
        # Process JSONL file
        count = 0
        with open(jsonl_path, "r") as f:
            for line in tqdm(f, desc=f"{split}"):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
                    continue
                
                idx = data.get("idx")
                if idx is None:
                    print(f"Warning: Skipping entry without idx")
                    continue
                
                # Convert modalities to tensors and write to LMDB
                for mod_key, mod_dir in MOD_DIRS.items():
                    if mod_key not in data:
                        continue
                    
                    mod_data = data[mod_key]
                    if mod_data is None or (isinstance(mod_data, list) and len(mod_data) == 0):
                        continue
                    
                    try:
                        if mod_key == "fragidx":
                            # FragIdx: list of ints -> int32 tensor (1D)
                            tensor = torch.tensor(mod_data, dtype=torch.int32)
                        elif mod_key in ["h_nmr", "c_nmr"]:
                            # 1D: list of floats -> 1D tensor (stored as 1D, reshaped in loader)
                            tensor = torch.tensor(mod_data, dtype=torch.float32)
                        elif mod_key in ["hsqc", "cosy", "hmbc", "mass_spec"]:
                            # 2D: list of [x, y] pairs -> (N, 2) tensor
                            arr = np.array(mod_data, dtype=np.float32)
                            if arr.ndim == 1:
                                # Handle edge case: single value
                                arr = arr.reshape(-1, 1)
                            tensor = torch.from_numpy(arr)
                        else:
                            # Fallback: try to convert to tensor
                            tensor = list_to_tensor(mod_data)
                        
                        split_writers[split][mod_key].add(idx, tensor)
                    except Exception as e:
                        print(f"Warning: Failed to convert {mod_key} for idx {idx}: {e}")
                
                count += 1
        
        # Close all writers for this split
        for writer in split_writers[split].values():
            writer.close()
        
        print(f"  Converted {count} entries for {split}")
    
    print("\nConversion complete!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert between LMDB and JSONL dataset formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert LMDB dataset to JSONL
  python convert.py --from-lmdb /path/to/lmdb_dataset /path/to/jsonl_dataset
  
  # Convert JSONL dataset to LMDB
  python convert.py --to-lmdb /path/to/jsonl_dataset /path/to/lmdb_dataset
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--from-lmdb",
        action="store_true",
        help="Convert from LMDB format to JSONL format"
    )
    group.add_argument(
        "--to-lmdb",
        action="store_true",
        help="Convert from JSONL format to LMDB format"
    )
    
    parser.add_argument(
        "input_dir",
        help="Input dataset directory"
    )
    parser.add_argument(
        "output_dir",
        help="Output dataset directory"
    )
    
    args = parser.parse_args()
    
    if args.from_lmdb:
        convert_from_lmdb(args.input_dir, args.output_dir)
    elif args.to_lmdb:
        convert_to_lmdb(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
