#!/usr/bin/env python
import os
import io
import pickle
import argparse

import lmdb
import torch
import numpy as np

# Splits and per-modality folder names (on disk)
SPLITS = ("train", "val", "test")
MOD_DIRS = {
    'hsqc': 'HSQC_NMR',
    'h_nmr': 'H_NMR',
    'c_nmr': 'C_NMR',
    'mass_spec': 'MassSpec',
}
FRAGMENTS_DIR = "Fragments"  # (used only as source to build FragIdx)

# ---------------- helpers ----------------

def _load_index(root):
    with open(os.path.join(root, 'index.pkl'), 'rb') as f:
        return pickle.load(f)

def _iter_split_items(index: dict, split: str):
    for idx, entry in index.items():
        if entry.get('split') == split:
            yield idx, entry

def _encode_array_raw(arr: np.ndarray) -> bytes:
    """Zero-copy friendly encoding: ASCII header + raw array bytes."""
    arr = np.ascontiguousarray(arr)
    header = f"{arr.dtype.name}|{arr.ndim}|{','.join(map(str, arr.shape))}|".encode("ascii")
    return header + memoryview(arr)

def _encode_tensor_raw(t: torch.Tensor) -> bytes:
    t = t.contiguous()
    arr = t.detach().cpu().numpy()
    return _encode_array_raw(arr)

def _estimate_mapsize(num_items: int, avg_bytes: int) -> int:
    # ~30% headroom + 256MB
    return int(max(1, num_items) * max(1, avg_bytes) * 1.3) + (256 << 20)

def _open_env_for_write(lmdb_path: str, mapsize: int):
    os.makedirs(lmdb_path, exist_ok=True)  # ensure dir exists when subdir=True
    return lmdb.open(
        lmdb_path,
        map_size=mapsize,
        subdir=True,
        lock=True,         # writer
        writemap=False,    # safer
        map_async=False,   # durability while building
        max_dbs=1,
    )

def _done_stats(env, lmdb_path: str, count: int):
    env.sync()
    info = env.info()
    stat = env.stat()
    data_file = os.path.join(lmdb_path, "data.mdb")
    size_gb = os.path.getsize(data_file)/1e9 if os.path.exists(data_file) else 0.0
    print(f"[done] {lmdb_path} wrote {count} entries "
          f"(entries={stat.get('entries')}, map_size={info.get('map_size')/1e9:.2f} GB, file≈{size_gb:.2f} GB)")
    env.close()

# ---------------- builders ----------------

def _build_lmdb_for_modality(root, out_dir, split, modality_key, modality_dir, items):
    """
    Build LMDB for a spectral modality (HSQC/H/C/MS) in raw header+bytes format.
    `items` must already be filtered to entries that have this modality (has_<modality_key>=True).
    """
    os.makedirs(out_dir, exist_ok=True)
    lmdb_path = os.path.join(out_dir, f"{modality_dir}.lmdb")
    if os.path.isdir(lmdb_path) and os.listdir(lmdb_path):
        print(f"[skip] {lmdb_path} already exists and is non-empty")
        return

    if not items:
        print(f"[warn] No items for {split}/{modality_key}; skipping")
        return

    sample_idx = items[0][0]
    sample_file = os.path.join(root, modality_dir, f"{sample_idx}.pt")
    sample_tensor = torch.load(sample_file, weights_only=True, map_location="cpu")
    sample_bytes = len(_encode_tensor_raw(sample_tensor))
    mapsize = _estimate_mapsize(len(items), sample_bytes)

    print(f"[create] {lmdb_path} map_size≈{mapsize/1e9:.2f} GB (n={len(items)}, avg≈{sample_bytes/1024:.1f} KB)")
    env = _open_env_for_write(lmdb_path, mapsize)

    count = 0
    batch = []
    N = len(items)

    for i, (idx, _entry) in enumerate(items, 1):
        src = os.path.join(root, modality_dir, f"{idx}.pt")
        t = torch.load(src, weights_only=True, map_location="cpu")
        buf = _encode_tensor_raw(t)
        batch.append((idx, buf))

        if (i % 2048) == 0 or i == N:
            try:
                with env.begin(write=True) as txn:
                    for k_idx, k_buf in batch:
                        txn.put(str(k_idx).encode("utf-8"), k_buf)
            except lmdb.MapFullError:
                env.set_mapsize(int(env.info()["map_size"] * 1.5))
                with env.begin(write=True) as txn:
                    for k_idx, k_buf in batch:
                        txn.put(str(k_idx).encode("utf-8"), k_buf)
            count += len(batch)
            print(f"  .. {i} items (wrote +{len(batch)}, total={count})")
            batch.clear()

    _done_stats(env, lmdb_path, count)

def _build_lmdb_for_fragidx(root, out_dir, split, items, bitinfo_to_idx_path: str):
    """
    Build LMDB for FragIdx: per-idx sorted unique int32 column indices (pre-mapped fingerprints).
    Requires a pickle mapping { BitInfo(tuple): int } created after entropy selection.
    """
    os.makedirs(out_dir, exist_ok=True)
    lmdb_path = os.path.join(out_dir, "FragIdx.lmdb")
    if os.path.isdir(lmdb_path) and os.listdir(lmdb_path):
        print(f"[skip] {lmdb_path} already exists and is non-empty")
        return

    # Load mapping BitInfo(tuple) -> col (int)
    with open(bitinfo_to_idx_path, "rb") as f:
        bitinfo_to_idx = pickle.load(f)
    get = bitinfo_to_idx.get

    # Only rows that have Fragments/{idx}.pt (source lists)
    items = [(idx, e) for idx, e in items if os.path.exists(os.path.join(root, FRAGMENTS_DIR, f"{idx}.pt"))]
    if not items:
        print(f"[info] No Fragments items in {split} to map into FragIdx")
        return

    # Estimate sample size
    first_idx = items[0][0]
    first_frags = torch.load(os.path.join(root, FRAGMENTS_DIR, f"{first_idx}.pt"),
                             map_location="cpu", weights_only=True)
    arr0 = np.asarray(sorted({get(b) for b in first_frags if get(b) is not None}), dtype=np.int32)
    sample_bytes = len(_encode_array_raw(arr0))
    mapsize = _estimate_mapsize(len(items), max(sample_bytes, 2048))

    print(f"[create] {lmdb_path} map_size≈{mapsize/1e9:.2f} GB (n={len(items)}, est≈{sample_bytes/1024:.1f} KB)")
    env = _open_env_for_write(lmdb_path, mapsize)

    count = 0
    batch = []
    N = len(items)

    for i, (idx, _entry) in enumerate(items, 1):
        src = os.path.join(root, FRAGMENTS_DIR, f"{idx}.pt")
        frag_list = torch.load(src, map_location="cpu", weights_only=True)  # iterable of BitInfo tuples
        cols = sorted({get(b) for b in frag_list if get(b) is not None})
        arr = np.asarray(cols, dtype=np.int32)
        buf = _encode_array_raw(arr)
        batch.append((idx, buf))

        if (i % 4096) == 0 or i == N:
            try:
                with env.begin(write=True) as txn:
                    for k_idx, k_buf in batch:
                        txn.put(str(k_idx).encode("utf-8"), k_buf)
            except lmdb.MapFullError:
                env.set_mapsize(int(env.info()["map_size"] * 1.5))
                with env.begin(write=True) as txn:
                    for k_idx, k_buf in batch:
                        txn.put(str(k_idx).encode("utf-8"), k_buf)
            count += len(batch)
            print(f"  .. {i} items (FragIdx +{len(batch)}, total={count})")
            batch.clear()

    _done_stats(env, lmdb_path, count)

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="Shard modalities to raw LMDB and build FragIdx.lmdb (new format only).")
    ap.add_argument("--root", required=True, help="DATASET_ROOT (contains index.pkl and modality folders)")
    ap.add_argument("--out", default=None, help="Output base for LMDB (_lmdb). Default: <root>/_lmdb")
    ap.add_argument("--split", default=None, choices=SPLITS + (None,), help="Optional single split to convert")
    ap.add_argument("--bitinfo-to-idx", required=True,
                    help="Path to bitinfo_to_idx.pkl (from entropy selection) to build FragIdx.lmdb.")
    args = ap.parse_args()

    root = args.root
    out_base = args.out or os.path.join(root, "_lmdb")
    index = _load_index(root)

    splits = [args.split] if args.split else SPLITS
    for split in splits:
        print(f"\n=== Split: {split} ===")
        items = [(idx, e) for idx, e in _iter_split_items(index, split)]
        if not items:
            print(f"[warn] No items in split {split}, skipping")
            continue

        out_dir = os.path.join(out_base, split)
        os.makedirs(out_dir, exist_ok=True)

        # 1) Spectral modalities (only those present)
        for key, mod_dir in MOD_DIRS.items():
            split_items = [(idx, e) for idx, e in items if e.get(f"has_{key}", False)]
            if not split_items:
                print(f"[info] No '{key}' items in {split}")
                continue
            _build_lmdb_for_modality(root, out_dir, split, key, mod_dir, split_items)

        # 2) NEW: FragIdx (fast path for training)
        _build_lmdb_for_fragidx(root, out_dir, split, items, bitinfo_to_idx_path=args.bitinfo_to_idx)

    print("\nAll done. Loaders expect raw LMDB tensors and FragIdx.lmdb.")

if __name__ == "__main__":
    main()
