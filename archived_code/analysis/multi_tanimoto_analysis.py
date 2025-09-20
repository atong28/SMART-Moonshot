# np_superclass_fp_accuracy_avg.py
# Same as original, but loads multiple checkpoints and averages their predicted probs.

from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import matplotlib.pyplot as plt

# -------- Project imports --------
from archived_code.self_attention.src.settings import Args
from archived_code.self_attention.src.fp_loaders import get_fp_loader
from archived_code.self_attention.src.dataset import MoonshotDataModule, collate
from archived_code.self_attention.src.model import build_model

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

OUTPUT_DIR = Path('results/analysis/np_superclass_fp_accuracy_avg')
BATCH_SIZE = 128
THRESH = 0.5
INCLUDE_MULTI = False
INCLUDE_UNKNOWN = False
MAKE_TOP20_BAR = True
TOP_K = 20

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

def get_np_superclass_info(dm, idx):
    meta = dm.test.data[idx][1]
    labels = meta.get('np_superclass', [])
    smiles = meta.get('smiles', '')
    if not labels:
        return 'unknown', None, smiles
    if len(labels) == 1:
        return 'single', labels[0], smiles
    return 'multi', ';'.join(labels), smiles

def load_model_from_ckpt(args, fp_loader, dm, ckpt_path, device):
    model = build_model(args, True, fp_loader, dm.combinations_names)
    model.setup_ranker()
    ensure_ranker_device(model.ranker, device)
    sd = torch.load(ckpt_path, weights_only=True)['state_dict']
    sd['embedding.weight'] = sd['NMR_type_embedding.weight']; del sd['NMR_type_embedding.weight']
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model

@torch.no_grad()
def eval_tanimoto_per_item(models, dm, batch_size=128, thresh=0.5, device=None,
                           include_multi=False, include_unknown=False):
    device = device or ('cuda:1' if torch.cuda.is_available() else 'cpu')
    per_item = []
    N = len(dm.test)
    for i in range(0, N, batch_size):
        j = min(i + batch_size, N)
        batch = [dm.test[k] for k in range(i, j)]
        inputs, labels, nmr_type_ind = collate(batch)

        # Move to device
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(device)
        elif isinstance(inputs, dict):
            for k2, v2 in inputs.items():
                if torch.is_tensor(v2):
                    inputs[k2] = v2.to(device)
        else:
            inputs = [t.to(device) if torch.is_tensor(t) else t for t in inputs]
        nmr_type_ind = nmr_type_ind.to(device)
        labels = labels.to(device)

        # Aggregate predictions across models
        probs_sum = 0
        for model in models:
            logits = model.forward(inputs, nmr_type_ind)
            probs_sum += torch.sigmoid(logits)
        probs_avg = probs_sum / len(models)

        preds = (probs_avg >= thresh).to(torch.uint8)

        # Labels binary
        labels_bin = (labels >= 0.5).to(torch.uint8) if labels.dtype.is_floating_point else labels.to(torch.uint8)

        # Tanimoto
        inter = (preds & labels_bin).sum(dim=1).to(torch.float32)
        a_sum = preds.sum(dim=1).to(torch.float32)
        b_sum = labels_bin.sum(dim=1).to(torch.float32)
        denom = a_sum + b_sum - inter + 1e-8
        tani  = (inter / denom).detach().cpu().numpy()

        a_sum_cpu = a_sum.cpu().numpy().astype(int)
        b_sum_cpu = b_sum.cpu().numpy().astype(int)

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

    return per_item

def summarize_by_class(per_item):
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
    stats.sort(key=lambda d: d["n"], reverse=True)
    order = [d["class"] for d in stats]
    return stats, order

def save_per_item_csv(per_item, path):
    with open(path, 'w') as f:
        f.write("idx,class,smiles,tanimoto,bits_pred,bits_true\n")
        for r in per_item:
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

# ======================
# Main
# ======================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    args = Args(**{'data_root': DATA_ROOT, 'input_types': INPUT_TYPES})
    fp_loader = get_fp_loader(args)
    dm = MoonshotDataModule(args, str(OUTPUT_DIR), fp_loader)
    dm.setup('fit'); dm.setup('test')

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    models = [load_model_from_ckpt(args, fp_loader, dm, ckpt, device) for ckpt in CKPTS]

    print("[INFO] Running fingerprint prediction + averaged Tanimoto on test split...")
    per_item = eval_tanimoto_per_item(
        models, dm,
        batch_size=BATCH_SIZE, thresh=THRESH, device=device,
        include_multi=INCLUDE_MULTI, include_unknown=INCLUDE_UNKNOWN
    )
    if not per_item:
        print("[WARN] No items collected after filtering — try INCLUDE_MULTI/INCLUDE_UNKNOWN=True.")
        return

    per_item_csv = OUTPUT_DIR / "per_item_tanimoto.csv"
    save_per_item_csv(per_item, per_item_csv)
    print(f"[INFO] Saved per-item → {per_item_csv}")

    stats, _ = summarize_by_class(per_item)
    per_class_csv = OUTPUT_DIR / "per_class_tanimoto.csv"
    save_per_class_csv(stats, per_class_csv)
    print(f"[INFO] Saved per-class → {per_class_csv}")

    if MAKE_TOP20_BAR and stats:
        bar_png = OUTPUT_DIR / "top20_mean_tanimoto_bar.png"
        plot_topk_bar(stats, TOP_K, bar_png,
                      title=f"Mean Tanimoto by NP superclass (Top-{min(TOP_K, len(stats))})")
        print(f"[INFO] Saved Top-{TOP_K} bar chart → {bar_png}")

    print("\nTop classes by count with mean tanimoto:")
    for d in stats[:min(10, len(stats))]:
        print(f'  {d["class"]:<30} n={d["n"]:<5} mean={d["mean"]:.3f}  median={d["median"]:.3f}')

if __name__ == "__main__":
    main()
