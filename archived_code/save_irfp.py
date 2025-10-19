import os, json, argparse, torch

def main(builder_out_dir: str, dtype: str, copy_meta: bool):
    dataset_root = builder_out_dir
    # 1) Load builder fingerprints
    fps = torch.load(os.path.join(builder_out_dir, "fingerprints.pt")).to(torch.float32)

    # 2) Row L2-normalize so data @ query == cosine
    norms = torch.linalg.norm(fps, dim=1, keepdim=True).clamp_min(1e-12)
    fps = fps / norms

    # 3) Optional storage dtype
    if dtype == "fp16":
        store = fps.half()
    elif dtype == "bf16":
        store = fps.bfloat16()
    else:
        store = fps  # fp32

    # 4) Save to DATASET_ROOT
    os.makedirs(dataset_root, exist_ok=True)
    torch.save(store, os.path.join(dataset_root, "rankingset.pt"))

    # 5) Copy SMILES index for your own lookups (ranker doesn't require it but it’s handy)
    for name in ("smiles_to_idx.json", "idx_to_smiles.json"):
        src = os.path.join(builder_out_dir, name)
        if os.path.exists(src):
            with open(src, "r") as f:
                obj = json.load(f)
            with open(os.path.join(dataset_root, name), "w") as f:
                json.dump(obj, f)

    # 6) Optionally copy meta/vocab (useful for downstream inspection)
    if copy_meta:
        for name in ("meta.pkl", "vocab.pkl"):
            src = os.path.join(builder_out_dir, name)
            if os.path.exists(src):
                dst = os.path.join(dataset_root, name)
                with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
                    fdst.write(fsrc.read())

    print(f"[ok] rankingset.pt saved to {os.path.join(dataset_root, 'rankingset.pt')}")
    print(f"[info] dtype stored: {dtype}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert IRFP builder outputs to SPECTRE rankingset.pt")
    p.add_argument("builder_out_dir", help="Directory produced by info_rich_fp_builder_exclusive.py")
    p.add_argument("--dtype", choices=["fp32","fp16","bf16"], default="fp32",
                   help="Storage dtype for rankingset.pt (ranker casts to fp32 at load)")
    p.add_argument("--copy_meta", action="store_true", help="Also copy meta.pkl and vocab.pkl")
    args = p.parse_args()
    main(args.builder_out_dir, args.dtype, args.copy_meta)
