#!/usr/bin/env python3
"""
Rank-1 accuracy per NP superclass (Top-20), using your cm(...) + RankingSet.

Outputs:
  - results/analysis/rank1_by_class_trial1/per_item_rank1.csv
  - results/analysis/rank1_by_class_trial1/per_class_rank1.csv
  - results/analysis/rank1_by_class_trial1/top20_rank1_bar.png
"""

from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
import matplotlib.pyplot as plt
import csv

# --- Your project imports ---
from archived_code.self_attention.src.settings import Args
from archived_code.self_attention.src.fp_loaders import get_fp_loader
from archived_code.self_attention.src.dataset import MoonshotDataModule, collate
from archived_code.self_attention.src.model import build_model
from archived_code.self_attention.src.ranker import RankingSet
from archived_code.self_attention.src.metrics import cm

# ======================
# CONFIG
# ======================
CKPT = 'old_ckpt/trial_1.ckpt'   # edit if needed
DATA_ROOT = '/data/nas-gpu/wang/atong/MoonshotDataset'
INPUT_TYPES = ['hsqc', 'h_nmr', 'c_nmr', 'mw']

OUTPUT_DIR = Path('results/analysis/rank1_by_class_trial1')
BATCH_SIZE = 128
TOP_K = 20
INCLUDE_MULTI = False
INCLUDE_UNKNOWN = False
SAVE_PNG = True
PNG_DPI = 260

# ======================
# Helpers
# ======================
def get_np_superclass(dm, idx):
    labels = dm.test.data[idx][1].get('np_superclass', [])
    if not labels:
        return 'unknown', None
    if len(labels) == 1:
        return 'single', labels[0]
    return 'multi', ';'.join(labels)

def load_model_and_data():
    args = Args(**{'data_root': DATA_ROOT, 'input_types': INPUT_TYPES})
    fp_loader = get_fp_loader(args)
    dm = MoonshotDataModule(args, str(OUTPUT_DIR), fp_loader)
    dm.setup('fit'); dm.setup('test')
    model = build_model(args, True, fp_loader, dm.combinations_names)
    model.setup_ranker()
    sd = torch.load(CKPT, weights_only=True)['state_dict']
    sd['embedding.weight'] = sd['NMR_type_embedding.weight']; del sd['NMR_type_embedding.weight']
    model.load_state_dict(sd, strict=True)
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()
    return model, dm, device

@torch.no_grad()
def collect_rank1_per_item(model, dm, device):
    ranker = model.ranker

    per_item = []  # rows: idx, class, rank1 (0/1)
    i, N = 0, len(dm.test)
    while i < N:
        j = min(i + BATCH_SIZE, N)
        batch = [dm.test[k] for k in range(i, j)]
        inputs, labels, nmr_type_ind = collate(batch)

        # device move
        device_inputs = None
        if isinstance(inputs, torch.Tensor):
            device_inputs = inputs.to(device)
        elif isinstance(inputs, dict):
            device_inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        else:
            device_inputs = [t.to(device) if torch.is_tensor(t) else t for t in inputs]
        labels = labels.to(device)
        nmr_type_ind = nmr_type_ind.to(device)

        # forward & loss (to mirror test_step)
        logits = model.forward(device_inputs, nmr_type_ind)
        loss = model.loss(logits, labels)

        # call cm(...)
        try:
            metrics, rank_res = cm(
                logits, labels, ranker, loss, model.loss,
                thresh=0.0,
                rank_by_soft_output=getattr(model, 'rank_by_soft_output', False),
                query_idx_in_rankingset=None,
                use_jaccard=getattr(model, 'use_jaccard', False),
                no_ranking=False
            )
        except TypeError:
            # fallback in case cm signature differs (older versions)
            metrics, rank_res = cm(
                logits, labels, ranker, loss, model.loss,
                thresh=0.0,
                rank_by_soft_output=getattr(model, 'rank_by_soft_output', False),
                use_jaccard=getattr(model, 'use_jaccard', False)
            )

        # rank_res can be boolean hits or integer ranks; normalize to 0/1 hit
        if isinstance(rank_res, torch.Tensor):
            rr = rank_res.detach().cpu()
            if rr.dtype == torch.bool:
                hits = rr.numpy().astype(int)
            else:
                # treat as numeric rank: hit if < 1 (strictly best)
                hits = (rr < 1).numpy().astype(int)
        else:
            # if cm returned list, assume already hit flags
            arr = np.array(rank_res)
            hits = arr.astype(int) if arr.dtype == np.bool_ else (arr < 1).astype(int)

        # collect per item
        for r, idx in enumerate(range(i, j)):
            ltype, lval = get_np_superclass(dm, idx)
            if ltype == 'single':
                cls = lval
            elif ltype == 'multi':
                if not INCLUDE_MULTI: continue
                cls = 'Multi'
            else:
                if not INCLUDE_UNKNOWN: continue
                cls = 'Unknown'
            per_item.append({"idx": idx, "class": cls, "rank1": int(hits[r])})
        i = j

    return per_item

def summarize_rank1(per_item):
    # class order by count
    counts = Counter([r['class'] for r in per_item])
    order = [c for c,_ in counts.most_common()]
    stats = []
    for c in order:
        vals = [r['rank1'] for r in per_item if r['class']==c]
        arr = np.asarray(vals, dtype=np.float32)
        stats.append({"class": c, "n": int(arr.size), "mean_rank1": float(arr.mean()) if arr.size else float('nan')})
    return stats

def save_csv_per_item(per_item, path):
    with open(path, 'w') as f:
        f.write("idx,class,rank1\n")
        for r in per_item:
            f.write(f"{r['idx']},{r['class']},{r['rank1']}\n")

def save_csv_per_class(stats, path):
    with open(path, 'w') as f:
        f.write("class,n,mean_rank1\n")
        for s in stats:
            f.write(f"{s['class']},{s['n']},{s['mean_rank1']:.6f}\n")

def plot_top20_rank1(stats, top_k, out_png, dpi=260):
    # Pick top_k by count (stats are already sorted by count)
    top = stats[:top_k]
    # Sort by mean_rank1 desc for display
    top = sorted(top, key=lambda d: d['mean_rank1'], reverse=True)
    labels = [d['class'] for d in top]
    vals = [d['mean_rank1'] for d in top]
    ns = [d['n'] for d in top]

    plt.figure(figsize=(10, max(4, 0.45*len(top))))
    y = np.arange(len(top))
    plt.barh(y, vals)
    for i, (v, n) in enumerate(zip(vals, ns)):
        plt.text(v + 0.01, i, f"{v:.3f} (n={n})", va='center', fontsize=8)
    plt.yticks(y, labels, fontsize=8)
    plt.gca().invert_yaxis()
    plt.xlim(0, 1.0)
    plt.xlabel("Mean Rank-1 Accuracy")
    plt.title(f"NP superclass — Mean Rank-1 (Top-{len(top)})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi, bbox_inches='tight')
    plt.close()

def write_web_csv(stats, out_path: Path):
    """
    stats: list of dicts with keys {'class','n','mean_rank1'}
    Writes columns required by the web app: superclass,value,n
    """
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["superclass", "value", "n"])
        for s in stats:
            w.writerow([s["class"], float(s["mean_rank1"]), int(s["n"])])
    print(f"[INFO] Wrote web CSV → {out_path.resolve()}")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model, dm, device = load_model_and_data()

    print("[INFO] Evaluating Rank-1 per item…")
    per_item = collect_rank1_per_item(model, dm, device)
    if not per_item:
        print("[WARN] No items found (check INCLUDE_MULTI/INCLUDE_UNKNOWN).")
        return

    # CSVs
    per_item_csv = OUTPUT_DIR / "per_item_rank1.csv"
    save_csv_per_item(per_item, per_item_csv)
    stats = summarize_rank1(per_item)
    per_class_csv = OUTPUT_DIR / "per_class_rank1.csv"
    save_csv_per_class(stats, per_class_csv)
    write_web_csv(stats, Path("./acc_rank1.csv"))
    print(f"[INFO] Saved: {per_item_csv}")
    print(f"[INFO] Saved: {per_class_csv}")

    # Plot Top-20 (by count)
    if SAVE_PNG and stats:
        top = min(TOP_K, len(stats))
        png = OUTPUT_DIR / f"top{top}_rank1_bar.png"
        plot_top20_rank1(stats, top, png, dpi=PNG_DPI)
        print(f"[INFO] Saved: {png}")

if __name__ == "__main__":
    main()
