#!/usr/bin/env python3
"""
Mean Rank per NP superclass (Top-20), averaged across multiple checkpoints.

Outputs:
  - results/analysis/meanrank_by_class_ensemble/per_item_meanrank_ensemble.csv
  - results/analysis/meanrank_by_class_ensemble/per_class_meanrank_ensemble.csv
  - results/analysis/meanrank_by_class_ensemble/top20_meanrank_bar.png
"""

from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
import matplotlib.pyplot as plt

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
CKPTS = [
    'old_ckpt/trial_1.ckpt',
    'old_ckpt/trial_2.ckpt',
    'old_ckpt/trial_3.ckpt',
]
DATA_ROOT = '/data/nas-gpu/wang/atong/MoonshotDataset'
INPUT_TYPES = ['hsqc', 'h_nmr', 'c_nmr', 'mw']

OUTPUT_DIR = Path('results/analysis/meanrank_by_class_ensemble')
BATCH_SIZE = 128
TOP_K = 20
INCLUDE_MULTI = False
INCLUDE_UNKNOWN = False
SAVE_PNG = True
PNG_DPI = 260

# ======================
# Helpers
# ======================
def ensure_ranker_device(ranker, device: torch.device | str):
    """Move all tensor fields inside RankingSet to the given device."""
    dev = torch.device(device)
    tensor_attr_names = [
        "data", "data_csr", "data_t", "norms",
        "fp_labels", "labels", "mw", "mw_norms",
        # add any other tensor fields your RankingSet carries
    ]
    for name in tensor_attr_names:
        t = getattr(ranker, name, None)
        if t is None:
            continue
        if torch.is_tensor(t):
            if t.device != dev:
                setattr(ranker, name, t.to(dev))
        # If you keep lists of tensors, move each
        elif isinstance(t, (list, tuple)) and t and torch.is_tensor(t[0]):
            moved = [ti.to(dev) for ti in t]
            setattr(ranker, name, type(t)(moved))
    # some codebases store a .device attribute; keep it in sync
    try:
        ranker.device = dev
    except Exception:
        pass
    return ranker

def get_np_superclass(dm, idx):
    labels = dm.test.data[idx][1].get('np_superclass', [])
    if not labels:
        return 'unknown', None
    if len(labels) == 1:
        return 'single', labels[0]
    return 'multi', ';'.join(labels)

def build_dm():
    args = Args(**{'data_root': DATA_ROOT, 'input_types': INPUT_TYPES})
    fp_loader = get_fp_loader(args)
    dm = MoonshotDataModule(args, str(OUTPUT_DIR), fp_loader)
    dm.setup('fit'); dm.setup('test')
    return args, fp_loader, dm

def load_model(args, fp_loader, dm, ckpt_path, device):
    model = build_model(args, True, fp_loader, dm.combinations_names)
    sd = torch.load(ckpt_path, weights_only=True)['state_dict']
    sd['embedding.weight'] = sd['NMR_type_embedding.weight']; del sd['NMR_type_embedding.weight']
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model

@torch.no_grad()
def collect_meanrank_per_item_single_model(model, dm, device):
    if not (hasattr(model, 'ranker') and model.ranker is not None):
        raise RuntimeError("No model.ranker found. Please attach a RankingSet to the model before running.")

    per_item = []
    i, N = 0, len(dm.test)
    while i < N:
        j = min(i + BATCH_SIZE, N)
        batch = [dm.test[k] for k in range(i, j)]
        inputs, labels, nmr_type_ind = collate(batch)

        # device move
        if isinstance(inputs, torch.Tensor):
            device_inputs = inputs.to(device)
        elif isinstance(inputs, dict):
            device_inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        else:
            device_inputs = [t.to(device) if torch.is_tensor(t) else t for t in inputs]
        labels = labels.to(device)
        nmr_type_ind = nmr_type_ind.to(device)

        logits = model.forward(device_inputs, nmr_type_ind)
        loss = model.loss(logits, labels)

        try:
            metrics, rank_res = cm(
                logits, labels, model.ranker, loss, model.loss,
                thresh=0.0,
                rank_by_soft_output=getattr(model, 'rank_by_soft_output', False),
                query_idx_in_rankingset=None,
                use_jaccard=getattr(model, 'use_jaccard', False),
                no_ranking=False
            )
        except TypeError:
            metrics, rank_res = cm(
                logits, labels, model.ranker, loss, model.loss,
                thresh=0.0,
                rank_by_soft_output=getattr(model, 'rank_by_soft_output', False),
                use_jaccard=getattr(model, 'use_jaccard', False)
            )

        # we want numeric per-sample ranks; if boolean hits were returned, fallback to batch mean
        if isinstance(rank_res, torch.Tensor):
            rr = rank_res.detach().cpu()
            if rr.dtype == torch.bool:
                per_sample = np.full(rr.shape[0], float(metrics['mean_rank']), dtype=np.float32)
            else:
                per_sample = rr.numpy().astype(np.float32)
        else:
            arr = np.array(rank_res)
            if arr.dtype == np.bool_:
                per_sample = np.full(arr.shape[0], float(metrics['mean_rank']), dtype=np.float32)
            else:
                per_sample = arr.astype(np.float32)

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
            per_item.append({"idx": idx, "class": cls, "mean_rank": float(per_sample[r])})
        i = j

    return per_item

def average_per_item_across_models(per_item_lists):
    """
    Merge by (idx, class), average mean_rank across models.
    Returns list of dicts: {idx, class, mean_rank_mean}.
    """
    bucket = defaultdict(list)
    for lst in per_item_lists:
        for row in lst:
            bucket[(row["idx"], row["class"])].append(row["mean_rank"])
    out = []
    for (idx, cls), vals in bucket.items():
        out.append({"idx": idx, "class": cls, "mean_rank_mean": float(np.mean(vals))})
    return out

def summarize_meanrank(per_item_avg):
    counts = Counter([r['class'] for r in per_item_avg])
    order = [c for c,_ in counts.most_common()]
    stats = []
    for c in order:
        vals = [r['mean_rank_mean'] for r in per_item_avg if r['class']==c]
        arr = np.asarray(vals, dtype=np.float32)
        stats.append({
            "class": c,
            "n": int(arr.size),
            "mean_meanrank": float(arr.mean()) if arr.size else float('nan'),
            "median": float(np.median(arr)) if arr.size else float('nan'),
            "p25": float(np.percentile(arr, 25)) if arr.size else float('nan'),
            "p75": float(np.percentile(arr, 75)) if arr.size else float('nan'),
        })
    return stats

def save_csv_per_item(per_item_avg, path):
    with open(path, 'w') as f:
        f.write("idx,class,mean_rank_mean\n")
        for r in per_item_avg:
            f.write(f"{r['idx']},{r['class']},{r['mean_rank_mean']:.6f}\n")

def save_csv_per_class(stats, path):
    with open(path, 'w') as f:
        f.write("class,n,mean_meanrank,median,p25,p75\n")
        for s in stats:
            f.write(f"{s['class']},{s['n']},{s['mean_meanrank']:.6f},{s['median']:.6f},{s['p25']:.6f},{s['p75']:.6f}\n")

def plot_top20_meanrank(stats, top_k, out_png, dpi=260):
    top = stats[:top_k]  # by count
    top = sorted(top, key=lambda d: d['mean_meanrank'])  # lower is better
    labels = [d['class'] for d in top]
    vals = [d['mean_meanrank'] for d in top]
    ns = [d['n'] for d in top]

    plt.figure(figsize=(10, max(4, 0.45*len(top))))
    y = np.arange(len(top))
    plt.barh(y, vals)
    for i, (v, n) in enumerate(zip(vals, ns)):
        plt.text(v + 0.02, i, f"{v:.2f} (n={n})", va='center', fontsize=8)
    plt.yticks(y, labels, fontsize=8)
    plt.gca().invert_yaxis()
    plt.xlabel("Mean of per-sample rank (averaged across models; lower is better)")
    plt.title(f"NP superclass — Mean Rank (Top-{len(top)})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi, bbox_inches='tight')
    plt.close()

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    args, fp_loader, dm = build_dm()

    per_model_lists = []
    for ck in CKPTS:
        print(f"[INFO] Evaluating {ck} …")
        model = load_model(args, fp_loader, dm, ck, device)
        model.setup_ranker()
        ensure_ranker_device(model.ranker, device)
        per_item = collect_meanrank_per_item_single_model(model, dm, device)
        per_model_lists.append(per_item)

    per_item_avg = average_per_item_across_models(per_model_lists)
    if not per_item_avg:
        print("[WARN] No items after filtering.")
        return

    per_item_csv = OUTPUT_DIR / "per_item_meanrank_ensemble.csv"
    save_csv_per_item(per_item_avg, per_item_csv)

    stats = summarize_meanrank(per_item_avg)
    per_class_csv = OUTPUT_DIR / "per_class_meanrank_ensemble.csv"
    save_csv_per_class(stats, per_class_csv)
    print(f"[INFO] Saved: {per_item_csv}")
    print(f"[INFO] Saved: {per_class_csv}")

    if SAVE_PNG and stats:
        top = min(TOP_K, len(stats))
        png = OUTPUT_DIR / f"top{top}_meanrank_bar.png"
        plot_top20_meanrank(stats, top, png, dpi=PNG_DPI)
        print(f"[INFO] Saved: {png}")

if __name__ == "__main__":
    main()
