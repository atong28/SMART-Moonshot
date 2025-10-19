import os, argparse, torch, re

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def row_l2(x: torch.Tensor, eps=1e-12) -> torch.Tensor:
    n = torch.linalg.norm(x).clamp_min(eps)
    return x / n

def main(dataset_root: str, fp_type: str, no_normalize: bool):
    root   = os.path.join(dataset_root, fp_type)
    fp_dir = os.path.join(root, "fp")
    assert os.path.isdir(fp_dir), f"Missing folder: {fp_dir}"

    # Discover all per-idx rows
    names = [n for n in os.listdir(fp_dir) if n.endswith(".pt")]
    if not names:
        raise RuntimeError(f"No .pt rows found in {fp_dir}")
    names.sort(key=natural_key)  # stable, numeric-friendly order

    # Infer D from the first row
    sample = torch.load(os.path.join(fp_dir, names[0]), map_location="cpu", weights_only=True)
    D = int(sample.numel())

    crow = [0]
    col_chunks, val_chunks = [], []
    nnz = 0
    N = len(names)

    for k, fname in enumerate(names, 1):
        v = torch.load(os.path.join(fp_dir, fname), map_location="cpu", weights_only=True).to(torch.float32)
        if not no_normalize:
            v = row_l2(v)  # exact row-L2 normalization as used for cosine rankingset
        nz = torch.nonzero(v, as_tuple=False).flatten()
        if nz.numel() > 0:
            col_chunks.append(nz.to(torch.int64))
            val_chunks.append(v[nz].to(torch.float32))  # keep fp32 for exactness
            nnz += nz.numel()
        crow.append(nnz)
        if (k % 20000) == 0:
            print(f"  processed {k}/{N} rows…")

    # Pack into CSR tensor
    crow_t = torch.tensor(crow, dtype=torch.int64)
    col_t  = torch.cat(col_chunks) if col_chunks else torch.zeros(0, dtype=torch.int64)
    val_t  = torch.cat(val_chunks) if val_chunks else torch.zeros(0, dtype=torch.float32)

    csr = torch.sparse_csr_tensor(crow_t, col_t, val_t, size=(N, D), dtype=torch.float32)

    # Save as a tensor (not a dict)
    out_path = os.path.join(root, "rankingset.pt")
    torch.save(csr, out_path)

    density = (nnz / (N * D)) if (N * D) > 0 else 0.0
    print(f"[ok] wrote {out_path}")
    print(f"      size=({N}, {D})  nnz={nnz}  density={density:.6g}  layout={csr.layout}")

if __name__ == "__main__":
    dataset_root = '/data/nas-gpu/wang/atong/MoonshotDatasetv2'
    fp_types = ["RankingBalanced", "RankingGlobal", "RankingSuperclass"]
    for fp_type in fp_types:
        main(dataset_root, fp_type, True)
