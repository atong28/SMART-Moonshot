import os
import pickle
import lmdb
import selfies as sf

DATASET_ROOT = os.getenv("DATASET_ROOT", "/data")  # or set your dataset root path here

def build_selfies_string_lmdb(dataset_root: str, split: str, map_size_bytes: int = 5_000_000_000):
    index_path = os.path.join(dataset_root, "index.pkl")
    with open(index_path, "rb") as f:
        index = pickle.load(f)

    out_dir = os.path.join(dataset_root, "_lmdb", split, "SELFIES.lmdb")
    os.makedirs(out_dir, exist_ok=True)

    env = lmdb.open(out_dir, map_size=map_size_bytes, subdir=True)
    written = 0
    skipped = 0

    txn = env.begin(write=True)
    try:
        for idx, entry in index.items():
            if entry.get("split") != split:
                continue
            smi = entry.get("smiles")
            if not smi:
                skipped += 1
                continue

            try:
                s = sf.encoder(smi)   # SMILES -> SELFIES
            except Exception:
                skipped += 1
                continue

            txn.put(str(int(idx)).encode("utf-8"), s.encode("utf-8"))
            written += 1

            if written % 5000 == 0:
                txn.commit()
                txn = env.begin(write=True)
    finally:
        txn.commit()
        env.sync()
        env.close()

    print(f"[{split}] wrote={written} skipped={skipped} -> {out_dir}")

if __name__ == "__main__":
    for split in ("train", "val", "test"):
        build_selfies_string_lmdb(DATASET_ROOT, split)
