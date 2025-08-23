import os
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import matplotlib.pyplot as plt

from archived_code.self_attention.src.settings import Args
from archived_code.self_attention.src.fp_loaders import get_fp_loader
from archived_code.self_attention.src.dataset import MoonshotDataModule, collate
from archived_code.self_attention.src.model import build_model

import umap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import KernelDensity
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb

# ======================
# CONFIG — tweak here
# ======================
CKPT = 'old_ckpt/trial_1.ckpt'      # <— single model
DATA_ROOT = '/data/nas-gpu/wang/atong/MoonshotDataset'
INPUT_TYPES = ['hsqc', 'h_nmr', 'c_nmr', 'mw']

OUTPUT_DIR = Path('results/analysis/np_superclass_top20_trial1')
TOP_K_CLASSES = 20
BATCH_SIZE = 64

# UMAP
UMAP_NEIGHBORS = 20
UMAP_MIN_DIST = 0.08
UMAP_RANDOM_STATE = 42

# Density cloud appearance
GRID_BINS = 420                  # resolution of the shared grid
ALPHA = 0.75                     # higher = less faded (0.5–0.75 good range)
BANDWIDTH_SCALE = 0.4            # <1.0 = tighter; try 0.3–0.6
GAMMA = 0.90                     # emphasis for dense cores (0.6–0.9)

SAVE_FIG = True
SAVE_SIL_CSV = True

# ======================
# Helpers
# ======================
def get_np_superclass_label(datamodule, idx):
    labels = datamodule.test.data[idx][1].get('np_superclass', [])
    if not labels: return 'unknown', None
    if len(labels) == 1: return 'single', labels[0]
    return 'multi', ';'.join(labels)

def select_top_k_classes(y_single, k=20):
    counts = Counter(y_single)
    items = [(c, n) for c, n in counts.items() if n >= 2]
    items.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in items[:k]]

def load_model_from_ckpt(args, fp_loader, datamodule, ckpt_path, device=None):
    model = build_model(args, True, fp_loader, datamodule.combinations_names)
    sd = torch.load(ckpt_path, weights_only=True)['state_dict']
    # align renamed key as in your example
    sd['embedding.weight'] = sd['NMR_type_embedding.weight']; del sd['NMR_type_embedding.weight']
    model.load_state_dict(sd, strict=True)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    return model, device

def collect_test_cls_embeddings(model, datamodule, batch_size=64, device=None):
    model.eval()
    device = device or next(model.parameters()).device
    cls_all, y_types_vals = [], []
    with torch.no_grad():
        N = len(datamodule.test)
        i = 0
        while i < N:
            j = min(i + batch_size, N)
            batch = [datamodule.test[k] for k in range(i, j)]
            inputs, labels, nmr_type_ind = collate(batch)

            # move to device
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(device)
            elif isinstance(inputs, dict):
                for k2, v2 in inputs.items():
                    if torch.is_tensor(v2):
                        inputs[k2] = v2.to(device)
            else:
                inputs = [t.to(device) if torch.is_tensor(t) else t for t in inputs]
            nmr_type_ind = nmr_type_ind.to(device)

            seq_reps = model.forward(inputs, nmr_type_ind, return_representations=True)
            if isinstance(seq_reps, np.ndarray):
                seq_reps = torch.from_numpy(seq_reps)
            cls = seq_reps[:, 0, :].cpu().numpy()  # CLS token
            cls_all.append(cls)

            for k in range(i, j):
                y_types_vals.append(get_np_superclass_label(datamodule, k))
            i = j

    X = np.vstack(cls_all)  # (N, D)
    idx_single = np.array([i for i,(t,_) in enumerate(y_types_vals) if t=='single'])
    y_single = np.array([y_types_vals[i][1] for i in idx_single], dtype=object)
    return X, idx_single, y_single

class SharedUMAP:
    def __init__(self, n_neighbors=20, min_dist=0.08, random_state=42):
        self.scaler = StandardScaler()
        self.reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                                 n_components=2, random_state=random_state)
    def fit(self, X):
        Xs = self.scaler.fit_transform(X)
        self.reducer.fit(Xs)
        return self
    def transform(self, X):
        Xs = self.scaler.transform(X)
        return self.reducer.transform(Xs)

# KDE utilities (no seaborn required)
def _silverman_bandwidth(points):
    n = max(2, points.shape[0])
    std = np.std(points, axis=0, ddof=1)
    return max(1.06 * float(np.mean(std)) * n ** (-1/5), 1e-3)

def kde_on_grid(points, xlim, ylim, xbins=400, ybins=400, bandwidth=None, bandwidth_scale=1.0):
    if points.shape[0] < 2: return None
    xmin, xmax = xlim; ymin, ymax = ylim
    xs = np.linspace(xmin, xmax, xbins)
    ys = np.linspace(ymin, ymax, ybins)
    Xg, Yg = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([Xg.ravel(), Yg.ravel()])

    bw0 = bandwidth if bandwidth is not None else _silverman_bandwidth(points)
    bw = max(bw0 * bandwidth_scale, 1e-3)

    kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(points)
    log_d = kde.score_samples(grid_pts)
    d = np.exp(log_d).reshape(ybins, xbins)
    if d.size and d.max() > 0:
        d = d / d.max()  # normalize to [0,1]
    return d

def alpha_composite_rgba(base_rgba, src_rgb, src_alpha):
    """
    Safe Porter–Duff 'over' with masking to avoid 0/0.
    base_rgba : (H,W,4) float32 in [0,1]
    src_rgb   : (3,) or (H,W,3) float32 in [0,1]
    src_alpha : (H,W) float32 in [0,1]
    """
    dst_rgb = base_rgba[..., :3]
    dst_a   = base_rgba[..., 3]

    if src_rgb.ndim == 1:
        src_rgb = np.broadcast_to(np.array(src_rgb, dtype=np.float32)[None, None, :], dst_rgb.shape)
    src_a = np.asarray(src_alpha, dtype=np.float32)

    # Optional: prune tiny alphas to prevent numerical fuzz & warnings
    src_a = np.where(src_a < 1e-6, 0.0, src_a)

    out_a = src_a + dst_a * (1.0 - src_a)

    # numerator for straight-alpha RGB
    numer = src_rgb * src_a[..., None] + dst_rgb * dst_a[..., None] * (1.0 - src_a[..., None])

    # Compute out_rgb only where out_a > 0 to avoid 0/0
    out_rgb = dst_rgb.copy()
    mask = out_a > 0.0
    np.divide(numer, out_a[..., None], out=out_rgb, where=mask[..., None])

    base_rgba[..., :3] = out_rgb
    base_rgba[..., 3]  = out_a
    return base_rgba

def plot_density_clouds_single(Z, labels, classes, out_png=None,
                               xbins=400, ybins=400, alpha=0.65, gamma=0.75,
                               bandwidth=None, bandwidth_scale=0.4, palette="tab20"):
    # bounds
    xmin, xmax = Z[:,0].min(), Z[:,0].max()
    ymin, ymax = Z[:,1].min(), Z[:,1].max()
    xlim, ylim = (xmin, xmax), (ymin, ymax)

    # colors
    if palette == "tab20":
        cmap = plt.get_cmap("tab20")
        colors = [np.array(cmap(i)[:3], dtype=np.float32) for i in range(len(classes))]
    else:
        base = plt.get_cmap("hsv")
        colors = [to_rgb(base(i/len(classes))) for i in range(len(classes))]

    H, W = ybins, xbins
    rgba = np.zeros((H, W, 4), dtype=np.float32)
    legend_handles = []

    # draw each class as a KDE layer
    for ci, cls in enumerate(classes):
        pts = Z[labels == cls]
        if pts.shape[0] < 2:  # need at least 2 points for KDE
            continue
        dens = kde_on_grid(pts, xlim, ylim, xbins=xbins, ybins=ybins,
                           bandwidth=bandwidth, bandwidth_scale=bandwidth_scale)
        if dens is None: 
            continue
        a = ALPHA * np.clip(dens, 0, 1) ** GAMMA
        a[a < 1e-5] = 0.0  # drop near-zero alpha to avoid haze and 0/0 cases
        rgba = alpha_composite_rgba(rgba, colors[ci], a)
        legend_handles.append(Patch(facecolor=colors[ci], edgecolor='none', label=str(cls)))

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    # slightly off-white background helps saturation
    ax.set_facecolor('#f6f6f6')
    fig.patch.set_facecolor('#f6f6f6')
    ax.imshow(rgba, origin='lower', extent=[xlim[0], xlim[1], ylim[0], ylim[1]], aspect='equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Top-20 NP superclasses — UMAP density clouds (Trial 1)", fontsize=12, pad=10)
    if legend_handles:
        ncol = 2 if max(len(str(c)) for c in classes) > 14 else 1
        ax.legend(handles=legend_handles, loc='upper right', fontsize=8, frameon=True, ncol=ncol)
    plt.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=220, bbox_inches='tight')
    return fig

def plot_points_umap_single(
    Z, labels, classes, out_png=None, out_pdf=None,
    point_size=6.0, point_alpha=0.7, max_per_class=None,
    palette="tab20", shuffle=True, rasterized_pdf=True, dpi=300
):
    """
    Scatter-only UMAP for Top-20 classes with tiny points.
    - Z: (N,2) array
    - labels: (N,) class names aligned to Z
    - classes: ordered list of class names to plot & legend
    """
    cmap = plt.get_cmap("tab20")
    colors = [np.array(cmap(i)[:3], dtype=np.float32) for i in range(len(classes))]

    # Build per-class views (with optional cap & shuffle)
    idx_by_class = []
    rng = np.random.RandomState(0)
    for cls in classes:
        m = (labels == cls)
        idx = np.where(m)[0]
        if shuffle:
            rng.shuffle(idx)
        if (max_per_class is not None) and (idx.size > max_per_class):
            idx = idx[:max_per_class]
        idx_by_class.append(idx)

    # Interleave classes so layering is fair (round-robin over class lists)
    interleaved = []
    lengths = [len(ix) for ix in idx_by_class]
    k = 0
    while True:
        added = False
        for j, ix in enumerate(idx_by_class):
            if k < len(ix):
                interleaved.append((j, ix[k]))
                added = True
        if not added:
            break
        k += 1

    fig, ax = plt.subplots(figsize=(8.8, 7.6))
    ax.set_facecolor("white")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Top-20 NP superclasses — UMAP points (Trial 1)", fontsize=12, pad=10)

    # Draw in chunks for speed
    # group back by class but preserve the interleaving order via a stable pass
    for j, cls in enumerate(classes):
        pts = np.array([Z[i] for c,i in interleaved if c == j])
        if pts.size == 0:
            continue
        ax.scatter(
            pts[:,0], pts[:,1],
            s=point_size, alpha=point_alpha,
            c=[colors[j]], edgecolors="none",
            linewidths=0, rasterized=rasterized_pdf  # has effect only in PDF backends
        )

    # Legend
    handles = [Patch(facecolor=colors[j], edgecolor='none', label=str(cls)) for j, cls in enumerate(classes)]
    ncol = 2 if max(len(str(c)) for c in classes) > 14 else 1
    ax.legend(handles=handles, loc='upper right', fontsize=8, frameon=True, ncol=ncol)
    plt.tight_layout()

    if out_png:
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight")  # rasterized points keep file light but sharp
    return fig


def per_class_silhouette(X, labels):
    counts = Counter(labels)
    keep = [c for c, n in counts.items() if n >= 2]
    if not keep: return {}
    mask = np.isin(labels, keep)
    Xf = X[mask]
    yf = np.array(labels)[mask]
    if len(set(yf)) < 2 or len(Xf) < 3:
        return {}
    s = silhouette_samples(Xf, yf, metric='euclidean')
    out = {}
    for cls in keep:
        m = (yf == cls)
        out[cls] = float(s[m].mean())
    return out

# ======================
# Main
# ======================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Data
    args = Args(**{'data_root': DATA_ROOT, 'input_types': INPUT_TYPES})
    fp_loader = get_fp_loader(args)
    dm = MoonshotDataModule(args, str(OUTPUT_DIR), fp_loader)
    dm.setup('test')

    # Model
    model, device = load_model_from_ckpt(args, fp_loader, dm, CKPT)

    # Extract CLS embeddings
    X_all, idx_single, y_single = collect_test_cls_embeddings(model, dm, batch_size=BATCH_SIZE, device=device)

    # Top-20 by count
    top20 = select_top_k_classes(y_single, k=TOP_K_CLASSES)
    print(f"[INFO] Selected Top-{len(top20)} classes:", top20)

    # Restrict to Top-20 single-label items
    mask_top = np.isin(y_single, top20)
    X = X_all[idx_single[mask_top]]
    y = y_single[mask_top]

    # UMAP fit/transform
    reducer = SharedUMAP(n_neighbors=UMAP_NEIGHBORS, min_dist=UMAP_MIN_DIST, random_state=UMAP_RANDOM_STATE).fit(X)
    Z = reducer.transform(X)

    # Plot density clouds
    png_path = OUTPUT_DIR / 'top20_umap_points_trial1.png' if SAVE_FIG else None
    pdf_path = OUTPUT_DIR / 'top20_umap_points_trial1.pdf' if SAVE_FIG else None

    plot_points_umap_single(
        Z, y, top20,
        out_png=png_path, out_pdf=pdf_path,
        point_size=10.0,          # ↓ to 2.0 for denser datasets; ↑ to 4.0 if too faint
        point_alpha=1.0,         # 0.6–0.8 is a good range
        max_per_class=None,      # e.g., 5000 to cap per-class points if huge
        palette="tab20",
        shuffle=True,
        rasterized_pdf=True,     # keeps vector text/legend; points rasterized for crispness/speed
        dpi=350                  # ↑ dpi for sharper PNGs
    )
    print(f"[INFO] Saved points figure → {png_path}")
    print(f"[INFO] Saved points PDF    → {pdf_path}")

    # Optional: silhouettes in CLS space (robust, not affected by 2-D)
    sil = per_class_silhouette(X, y)
    if SAVE_SIL_CSV and sil:
        out_csv = OUTPUT_DIR / 'per_class_silhouette_trial1.csv'
        with open(out_csv, 'w') as f:
            f.write("class,mean_silhouette\n")
            for cls, val in sorted(sil.items(), key=lambda kv: kv[1], reverse=True):
                f.write(f"{cls},{val:.6f}\n")
        print(f"[INFO] Saved silhouettes → {out_csv}")

if __name__ == "__main__":
    main()
