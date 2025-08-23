# np_superclass_fp_accuracy.py
# Evaluate model-predicted 16,384-bit fingerprints against ground truth on the test split.
# Outputs:
#   - per_item_tanimoto.csv  (idx, class, smiles, tanimoto, bits_pred, bits_true)
#   - per_class_tanimoto.csv (class, n, mean, std, median, p25, p75)
#   - (optional) top20_mean_tanimoto_bar.png

from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import torch
import matplotlib.pyplot as plt
import csv

# -------- Project imports (your repo) --------
from archived_code.self_attention.src.settings import Args
from archived_code.self_attention.src.fp_loaders import get_fp_loader
from archived_code.self_attention.src.dataset import MoonshotDataModule, collate
from archived_code.self_attention.src.model import build_model

# ======================
# CONFIG — tweak here
# ======================
CKPT = 'old_ckpt/trial_1.ckpt'          # model checkpoint
DATA_ROOT = '/data/nas-gpu/wang/atong/MoonshotDataset'
INPUT_TYPES = ['hsqc', 'h_nmr', 'c_nmr', 'mw']

OUTPUT_DIR = Path('results/analysis/np_superclass_fp_accuracy_trial1')
BATCH_SIZE = 128
THRESH = 0.5                             # sigmoid threshold for binary FP
INCLUDE_MULTI = False                    # include molecules with >=2 NP superclasses as their own "Multi" bucket
INCLUDE_UNKNOWN = False                  # include molecules with 0 NP superclasses as "Unknown"
MAKE_TOP20_BAR = True                    # save a bar chart of top-20 classes by count
TOP_K = 20

# ======================
# Helpers
# ======================
def get_np_superclass_info(dm, idx):
    """Return (label_type, label_value, smiles)
       label_type in {'single','multi','unknown'}
       label_value is str for 'single', ';'-joined for 'multi', or None for 'unknown'
    """
    meta = dm.test.data[idx][1]
    labels = meta.get('np_superclass', [])
    smiles = meta.get('smiles', '')
    if not labels:
        return 'unknown', None, smiles
    if len(labels) == 1:
        return 'single', labels[0], smiles
    return 'multi', ';'.join(labels), smiles

def load_model_from_ckpt(args, fp_loader, dm, ckpt_path, device=None):
    model = build_model(args, True, fp_loader, dm.combinations_names)
    sd = torch.load(ckpt_path, weights_only=True)['state_dict']
    # key rename per your earlier code
    sd['embedding.weight'] = sd['NMR_type_embedding.weight']; del sd['NMR_type_embedding.weight']
    model.load_state_dict(sd, strict=True)
    device = device or ('cuda:1' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    return model, device

@torch.no_grad()
def eval_tanimoto_per_item(model, dm, batch_size=128, thresh=0.5, device=None,
                           include_multi=False, include_unknown=False):
    device = device or next(model.parameters()).device
    per_item = []
    N = len(dm.test)
    i = 0
    while i < N:
        j = min(i + batch_size, N)
        batch = [dm.test[k] for k in range(i, j)]
        inputs, labels, nmr_type_ind = collate(batch)

        # ----- move EVERYTHING to same device -----
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(device)
        elif isinstance(inputs, dict):
            for k2, v2 in inputs.items():
                if torch.is_tensor(v2):
                    inputs[k2] = v2.to(device)
        else:
            inputs = [t.to(device) if torch.is_tensor(t) else t for t in inputs]
        nmr_type_ind = nmr_type_ind.to(device)
        labels = labels.to(device)  # <-- important

        # forward → logits → probs → preds
        logits = model.forward(inputs, nmr_type_ind)              # (B, 16384) on device
        probs  = torch.sigmoid(logits)
        preds  = (probs >= thresh).to(torch.uint8)

        # ground-truth labels → binary on SAME device
        if labels.dtype.is_floating_point:
            labels_bin = (labels >= 0.5).to(torch.uint8)
        else:
            labels_bin = labels.to(torch.uint8)

        # tanimoto per row (still on GPU)
        inter = (preds & labels_bin).sum(dim=1).to(torch.float32)
        a_sum = preds.sum(dim=1).to(torch.float32)
        b_sum = labels_bin.sum(dim=1).to(torch.float32)
        denom = a_sum + b_sum - inter + 1e-8
        tani  = (inter / denom).detach().cpu().numpy()

        # small scalars for logging
        a_sum_cpu = a_sum.detach().cpu().numpy().astype(int)
        b_sum_cpu = b_sum.detach().cpu().numpy().astype(int)

        # collect meta per item
        for r, idx in enumerate(range(i, j)):
            ltype, lval, smiles = get_np_superclass_info(dm, idx)
            if ltype == 'single':
                class_name = lval
            elif ltype == 'multi':
                if not include_multi: continue
                class_name = 'Multi'
            else:
                if not include_unknown: continue
                class_name = 'Unknown'

            per_item.append({
                "idx": idx,
                "class": class_name,
                "smiles": smiles,
                "tanimoto": float(tani[r]),
                "bits_pred": int(a_sum_cpu[r]),
                "bits_true": int(b_sum_cpu[r]),
            })

        i = j

    return per_item

def summarize_by_class(per_item):
    """Return per-class summary dicts and sorted class order by count."""
    buckets = defaultdict(list)
    for row in per_item:
        buckets[row["class"]].append(row["tanimoto"])
    stats = []
    for cls, vals in buckets.items():
        arr = np.array(vals, dtype=np.float32)
        stats.append({
            "class": cls,
            "n": int(arr.size),
            "mean": float(arr.mean()) if arr.size else float('nan'),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "median": float(np.median(arr)) if arr.size else float('nan'),
            "p25": float(np.percentile(arr, 25)) if arr.size else float('nan'),
            "p75": float(np.percentile(arr, 75)) if arr.size else float('nan'),
        })
    # sort by count desc
    stats.sort(key=lambda d: d["n"], reverse=True)
    order = [d["class"] for d in stats]
    return stats, order

def save_per_item_csv(per_item, path):
    with open(path, 'w') as f:
        f.write("idx,class,smiles,tanimoto,bits_pred,bits_true\n")
        for r in per_item:
            # escape commas in smiles if any by wrapping in quotes
            smi = r["smiles"]
            if ',' in smi or '"' in smi:
                smi = '"' + smi.replace('"', '""') + '"'
            f.write(f'{r["idx"]},{r["class"]},{smi},{r["tanimoto"]:.6f},{r["bits_pred"]},{r["bits_true"]}\n')

def save_per_class_csv(stats, path):
    with open(path, 'w') as f:
        f.write("class,n,mean,std,median,p25,p75\n")
        for d in stats:
            f.write(f'{d["class"]},{d["n"]},{d["mean"]:.6f},{d["std"]:.6f},{d["median"]:.6f},{d["p25"]:.6f},{d["p75"]:.6f}\n')

def plot_topk_bar(stats, top_k, out_png, dpi=220, title="Mean Tanimoto by NP superclass (Top-20)"):
    # pick top_k by n (already sorted), then sort those by mean descending for display
    top = stats[:top_k]
    top = sorted(top, key=lambda d: d["mean"], reverse=True)
    labels = [d["class"] for d in top]
    means = [d["mean"] for d in top]
    ns = [d["n"] for d in top]

    plt.figure(figsize=(10, max(4, 0.4*len(top))))
    y = np.arange(len(top))
    plt.barh(y, means)
    for i, (m, n) in enumerate(zip(means, ns)):
        plt.text(m + 0.005, i, f"{m:.3f} (n={n})", va='center', fontsize=8)
    plt.yticks(y, labels, fontsize=8)
    plt.gca().invert_yaxis()
    plt.xlabel("Mean Tanimoto (pred vs ground truth)")
    plt.title(title)
    plt.xlim(0, 1.0)
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi, bbox_inches='tight')
    plt.close()

def write_web_csv(stats, out_path: Path):
    """
    stats: list of dicts from summarize_by_class()
           keys include {'class','n','mean', ...}
    Writes columns required by the web app: superclass,value,n
    """
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["superclass", "value", "n"])
        for s in stats:
            w.writerow([s["class"], float(s["mean"]), int(s["n"])])
    print(f"[INFO] Wrote web CSV → {out_path.resolve()}")

# ======================
# Main
# ======================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Data + model
    args = Args(**{'data_root': DATA_ROOT, 'input_types': INPUT_TYPES})
    fp_loader = get_fp_loader(args)
    dm = MoonshotDataModule(args, str(OUTPUT_DIR), fp_loader)
    dm.setup('fit'); dm.setup('test')

    model, device = load_model_from_ckpt(args, fp_loader, dm, CKPT)

    # Evaluate
    print("[INFO] Running fingerprint prediction + Tanimoto on test split...")
    per_item = eval_tanimoto_per_item(
        model, dm,
        batch_size=BATCH_SIZE, thresh=THRESH, device=device,
        include_multi=INCLUDE_MULTI, include_unknown=INCLUDE_UNKNOWN
    )
    if not per_item:
        print("[WARN] No items collected after filtering — try setting INCLUDE_MULTI/INCLUDE_UNKNOWN=True.")
        return

    # Save per-item
    per_item_csv = OUTPUT_DIR / "per_item_tanimoto.csv"
    save_per_item_csv(per_item, per_item_csv)
    print(f"[INFO] Saved per-item results → {per_item_csv}")

    # Summarize per class
    stats, order = summarize_by_class(per_item)
    per_class_csv = OUTPUT_DIR / "per_class_tanimoto.csv"
    save_per_class_csv(stats, per_class_csv)
    write_web_csv(stats, Path("./acc_tanimoto.csv"))
    print(f"[INFO] Saved per-class summary → {per_class_csv}")

    # Optional bar chart
    if MAKE_TOP20_BAR and stats:
        bar_png = OUTPUT_DIR / "top20_mean_tanimoto_bar.png"
        plot_topk_bar(stats, TOP_K, bar_png, dpi=240,
                      title=f"Mean Tanimoto by NP superclass (Top-{min(TOP_K, len(stats))})")
        print(f"[INFO] Saved Top-{TOP_K} bar chart → {bar_png}")

    # Console preview
    print("\nTop classes by count with mean tanimoto:")
    for d in stats[:min(10, len(stats))]:
        print(f'  {d["class"]:<30} n={d["n"]:<5} mean={d["mean"]:.3f}  median={d["median"]:.3f}')

if __name__ == "__main__":
    main()
