import os
import json
import lmdb
import numpy as np
import selfies as sf

DATASET_ROOT = "/data"  # change if needed

def encode_ndarray(arr: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(arr)
    dtype_str = str(arr.dtype)
    shape_str = ",".join(str(d) for d in arr.shape) if arr.ndim > 0 else ""
    header = f"{dtype_str}|{arr.ndim}|{shape_str}|".encode("ascii")
    return header + arr.tobytes(order="C")

def selfies_to_ids(selfies_str: str, vocab: dict) -> np.ndarray:
    toks = list(sf.split_selfies(selfies_str))
    unk = vocab["[UNK]"]
    ids = [vocab.get(t, unk) for t in toks]
    ids = [vocab["[BOS]"]] + ids + [vocab["[EOS]"]]
    return np.asarray(ids, dtype=np.int32)

def convert_split(split: str, vocab: dict, map_size_bytes: int = 6_000_000_000):
    in_dir  = os.path.join(DATASET_ROOT, "_lmdb", split, "SELFIES.lmdb")
    out_dir = os.path.join(DATASET_ROOT, "_lmdb", split, "SELFIES_IDs.lmdb")
    os.makedirs(out_dir, exist_ok=True)

    env_in  = lmdb.open(in_dir,  readonly=True, lock=False, readahead=False, subdir=True, max_readers=4096)
    env_out = lmdb.open(out_dir, map_size=map_size_bytes, subdir=True)

    n = 0
    txn_out = env_out.begin(write=True)

    with env_in.begin(write=False, buffers=True) as txn_in:
        cur = txn_in.cursor()
        for k, v in cur:
            s = bytes(v).decode("utf-8")
            ids = selfies_to_ids(s, vocab)
            txn_out.put(bytes(k), encode_ndarray(ids))
            n += 1
            if n % 5000 == 0:
                txn_out.commit()
                txn_out = env_out.begin(write=True)

    txn_out.commit()
    env_out.sync()
    env_in.close()
    env_out.close()
    print(f"[{split}] wrote {n} -> {out_dir}")

if __name__ == "__main__":
    vocab_path = os.path.join(DATASET_ROOT, "selfies_vocab.json")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    for split in ("train", "val", "test"):
        convert_split(split, vocab)
