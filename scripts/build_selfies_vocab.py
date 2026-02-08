import os
import json
from collections import Counter
import lmdb

def split_selfies(s: str) -> list[str]:
    toks = []
    buf = ""
    for ch in s:
        buf += ch
        if ch == "]":
            toks.append(buf)
            buf = ""
    if buf:
        toks.append(buf)
    return toks

def iter_selfies_from_lmdb(lmdb_dir: str):
    env = lmdb.open(lmdb_dir, readonly=True, lock=False, readahead=False, subdir=True, max_readers=4096)
    with env.begin(write=False, buffers=True) as txn:
        cur = txn.cursor()
        for k, v in cur:
            # v is memoryview when buffers=True
            yield bytes(v).decode("utf-8")

def build_vocab_from_splits(
    dataset_root: str,
    splits=("train", "val", "test"),
    lmdb_name="SELFIES.lmdb",
    max_vocab=8192,
    min_freq=2,
):
    counter = Counter()
    n_strings = 0
    n_tokens = 0

    for split in splits:
        lmdb_dir = os.path.join(dataset_root, "_lmdb", split, lmdb_name)
        if not os.path.isdir(lmdb_dir):
            raise FileNotFoundError(f"Missing LMDB dir: {lmdb_dir}")

        for s in iter_selfies_from_lmdb(lmdb_dir):
            n_strings += 1
            toks = split_selfies(s)
            n_tokens += len(toks)
            counter.update(toks)

    # Special tokens (reserve IDs)
    vocab = {
        "[PAD]": 0,
        "[BOS]": 1,
        "[EOS]": 2,
        "[UNK]": 3,
    }

    # Add tokens by frequency
    for tok, freq in counter.most_common():
        if freq < min_freq:
            break
        if tok in vocab:
            continue
        vocab[tok] = len(vocab)
        if len(vocab) >= max_vocab:
            break

    stats = {
        "n_strings": n_strings,
        "avg_len_tokens": (n_tokens / n_strings) if n_strings else 0.0,
        "unique_tokens_seen": len(counter),
        "vocab_size": len(vocab),
        "min_freq": min_freq,
        "max_vocab": max_vocab,
        "top10": counter.most_common(10),
    }
    return vocab, stats, counter

if __name__ == "__main__":
    # CHANGE THIS to your dataset root (looks like /data in your output)
    DATASET_ROOT = os.getenv("DATASET_ROOT", "/data")

    vocab, stats, counter = build_vocab_from_splits(
        dataset_root=DATASET_ROOT,
        splits=("train", "val", "test"),
        lmdb_name="SELFIES.lmdb",
        max_vocab=8192,
        min_freq=2,
    )

    out_vocab = os.path.join(DATASET_ROOT, "selfies_vocab.json")
    out_stats = os.path.join(DATASET_ROOT, "selfies_vocab_stats.json")

    with open(out_vocab, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("Wrote:", out_vocab)
    print("Wrote:", out_stats)
    print("Stats:", stats)
