#!/usr/bin/env python3
"""
NP superclass compactness/separation in Morgan (2048) Tanimoto space + plots.

Outputs in OUTPUT_DIR:
  CSVs:
    - per_item_silhouette.csv
    - per_class_compactness.csv
    - class_class_mean_tanimoto.csv
  FIGs:
    - heatmap_class_class_tanimoto.png
    - bar_mean_intra_similarity.png
    - scatter_intra_vs_silhouette.png
"""

from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# ---- Project imports (your repo) ----
from archived_code.self_attention.src.settings import Args
from archived_code.self_attention.src.fp_loaders import get_fp_loader
from archived_code.self_attention.src.dataset import MoonshotDataModule
# model not needed

# ---- External deps ----
from sklearn.metrics import pairwise_distances, silhouette_samples
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs

# ======================
# CONFIG — tweak here
# ======================
DATA_ROOT = '/data/nas-gpu/wang/atong/MoonshotDataset'
INPUT_TYPES = ['hsqc', 'h_nmr', 'c_nmr', 'mw']

OUTPUT_DIR = Path('results/analysis/np_superclass_tanimoto_compactness')
TOP_K_CLASSES = 20           # analyze top-K classes by count (single-label only)
N_BITS = 2048
RADIUS = 2                   # 2 -> ECFP4, 3 -> ECFP6
SAVE_SILHOUETTE = True       # per-item silhouette CSV
MIN_CLASS_SIZE = 2           # skip classes with fewer than this many items

# Figure settings
DPI = 300
CMAP_HEATMAP = "viridis"

# ======================
# Helpers
# ======================
def get_np_superclass_label(dm, idx):
    labels = dm.test.data[idx][1].get('np_superclass', [])
    if not labels: return 'unknown', None
    if len(labels) == 1: return 'single', labels[0]
    return 'multi', ';'.join(labels)

def smiles_list_for_test(dm):
    return [dm.test.data[i][1].get('smiles', '') for i in range(len(dm.test))]

def select_top_k_classes(y_single, k=20, min_size=2):
    counts = Counter(y_single)
    items = [(c, n) for c, n in counts.items() if n >= min_size]
    items.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in items[:k]]

def morgan_fp_matrix(smiles_list, idx_subset, n_bits=2048, radius=2):
    """Return (F_bool, ok_mask) for subset indices -> boolean matrix (M, n_bits)."""
    M = len(idx_subset)
    F = np.zeros((M, n_bits), dtype=np.uint8)
    ok = np.zeros(M, dtype=bool)
    tmp = np.zeros((n_bits,), dtype=np.int8)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    for r, idx in enumerate(idx_subset):
        smi = smiles_list[idx]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        bv = gen.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(bv, tmp)
        F[r] = (tmp > 0).astype(np.uint8)
        ok[r] = True
    return F.astype(bool), ok

def medoid_index_jaccard(F_bool):
    """Return index of medoid under Jaccard distance (lower mean distance)."""
    if F_bool.shape[0] == 1:
        return 0
    D = pairwise_distances(F_bool, metric='jaccard')  # (n,n)
    np.fill_diagonal(D, 0.0)
    row_means = D.mean(axis=1)
    return int(np.argmin(row_means))

def intra_class_stats(F_bool):
    """
    Compute intra-class compactness stats for a boolean FP matrix of shape (n, d).
    Returns dict with:
      - mean_intra_similarity
      - median_intra_similarity
      - mean_similarity_to_medoid
      - mean_nearest_neighbor_similarity
      - medoid_idx
    """
    n = F_bool.shape[0]
    if n < 2:
        return {
            "mean_intra_similarity": float('nan'),
            "median_intra_similarity": float('nan'),
            "mean_similarity_to_medoid": float('nan'),
            "mean_nearest_neighbor_similarity": float('nan'),
            "medoid_idx": 0
        }

    # Pairwise Jaccard distance within class
    D = pairwise_distances(F_bool, metric='jaccard')  # 1 - Tanimoto
    np.fill_diagonal(D, np.nan)  # exclude self

    # Intra-class Tanimoto similarities from distances
    S = 1.0 - D
    iu = np.triu_indices(n, k=1)
    pair_sims = S[iu]
    pair_sims = pair_sims[~np.isnan(pair_sims)]

    # Nearest neighbor similarity per sample
    nn_sims = 1.0 - np.nanmin(D, axis=1)

    # Medoid & similarity to medoid
    med_idx = medoid_index_jaccard(F_bool)
    sims_to_medoid = 1.0 - D[med_idx, :]
    sims_to_medoid = sims_to_medoid[~np.isnan(sims_to_medoid)]

    return {
        "mean_intra_similarity": float(np.mean(pair_sims)) if pair_sims.size else float('nan'),
        "median_intra_similarity": float(np.median(pair_sims)) if pair_sims.size else float('nan'),
        "mean_similarity_to_medoid": float(np.mean(sims_to_medoid)) if sims_to_medoid.size else float('nan'),
        "mean_nearest_neighbor_similarity": float(np.mean(nn_sims)) if nn_sims.size else float('nan'),
        "medoid_idx": med_idx
    }

# ======================
# Plotting helpers
# ======================
def plot_heatmap(M, labels, out_path, title="Class–Class Medoid Tanimoto"):
    fig, ax = plt.subplots(figsize=(1.0 + 0.32*len(labels), 1.0 + 0.32*len(labels)))
    im = ax.imshow(M, cmap=CMAP_HEATMAP, vmin=0.0, vmax=1.0, origin='upper')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Tanimoto", rotation=90, va='center')
    ax.set_title(title, pad=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)

def plot_bar_mean_intra(per_class_stats, out_path, title="Mean Intra-class Tanimoto (Top-K by count)"):
    # sort by mean_intra_similarity desc for display
    stats = sorted(per_class_stats, key=lambda d: d["mean_intra_similarity"], reverse=True)
    labels = [d["class"] for d in stats]
    vals = [d["mean_intra_similarity"] for d in stats]
    ns = [d["n"] for d in stats]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.45*len(labels))))
    y = np.arange(len(labels))
    ax.barh(y, vals)
    for i, (v, n) in enumerate(zip(vals, ns)):
        ax.text(v + 0.01, i, f"{v:.3f} (n={n})", va='center', fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Intra-class Tanimoto")
    ax.set_xlim(0, 1.0)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)

def plot_scatter_intra_vs_silhouette(per_class_stats, out_path, title="Intra-class Tanimoto vs. Silhouette (Top-K)"):
    # keep only rows with valid silhouette_mean
    stats = [d for d in per_class_stats if not np.isnan(d["silhouette_mean"])]
    if not stats:
        return
    x = [d["mean_intra_similarity"] for d in stats]
    y = [d["silhouette_mean"] for d in stats]
    s = [max(20, 4*np.sqrt(d["n"])) for d in stats]  # size ~ sqrt(n)
    labels = [d["class"] for d in stats]

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.scatter(x, y, s=s, alpha=0.8)
    for xi, yi, lab in zip(x, y, labels):
        ax.annotate(lab, (xi, yi), textcoords="offset points", xytext=(4, 2), fontsize=8)
    ax.set_xlabel("Mean Intra-class Tanimoto (higher = tighter)")
    ax.set_ylabel("Mean Silhouette (higher = better separated)")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)

# ======================
# Main
# ======================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Data module (no model needed for FP compactness)
    args = Args(**{'data_root': DATA_ROOT, 'input_types': INPUT_TYPES})
    fp_loader = get_fp_loader(args)
    dm = MoonshotDataModule(args, str(OUTPUT_DIR), fp_loader)
    dm.setup('fit'); dm.setup('test')

    # Collect single-label class names
    y_types_vals = []
    for i in range(len(dm.test)):
        y_types_vals.append(get_np_superclass_label(dm, i))
    idx_single = np.array([i for i, (t, _) in enumerate(y_types_vals) if t == 'single'])
    y_single = np.array([y_types_vals[i][1] for i in idx_single], dtype=object)

    # Top-K classes
    topK = select_top_k_classes(y_single, k=TOP_K_CLASSES, min_size=MIN_CLASS_SIZE)
    print(f"[INFO] Selected Top-{len(topK)} classes:", topK)

    # Filter to Top-K single-label items
    mask_top = np.isin(y_single, topK)
    idx_top = idx_single[mask_top]
    y_top = y_single[mask_top]
    smiles = smiles_list_for_test(dm)

    # Build Morgan FPs for the Top-K subset
    F_bool, ok = morgan_fp_matrix(smiles, idx_top, n_bits=N_BITS, radius=RADIUS)
    if not np.all(ok):
        F_bool = F_bool[ok]
        y_top = y_top[ok]
        idx_top = idx_top[ok]

    # Per-class compactness & medoids
    per_class = []
    class_to_indices = {c: np.where(y_top == c)[0] for c in topK}
    medoids = {}

    # Silhouette across the whole Top-K subset (only for classes with >=2 members)
    valid_classes = [c for c in topK if class_to_indices[c].size >= 2]
    if SAVE_SILHOUETTE and len(valid_classes) >= 2:
        mask_valid = np.isin(y_top, valid_classes)
        sil_vals = silhouette_samples(F_bool[mask_valid], y_top[mask_valid], metric='jaccard')
        full_sil = np.full(F_bool.shape[0], np.nan, dtype=np.float32)
        full_sil[mask_valid] = sil_vals
    else:
        full_sil = np.full(F_bool.shape[0], np.nan, dtype=np.float32)

    for cls in topK:
        ix = class_to_indices[cls]
        if ix.size < MIN_CLASS_SIZE:
            continue
        F_c = F_bool[ix]
        stats = intra_class_stats(F_c)
        med_local = stats["medoid_idx"]
        med_global = ix[med_local]
        medoids[cls] = med_global

        sil_class = full_sil[ix]
        sil_clean = sil_class[~np.isnan(sil_class)]

        per_class.append({
            "class": cls,
            "n": int(ix.size),
            "mean_intra_similarity": stats["mean_intra_similarity"],
            "median_intra_similarity": stats["median_intra_similarity"],
            "mean_similarity_to_medoid": stats["mean_similarity_to_medoid"],
            "mean_nearest_neighbor_similarity": stats["mean_nearest_neighbor_similarity"],
            "silhouette_mean": float(np.mean(sil_clean)) if sil_clean.size else float('nan'),
            "silhouette_median": float(np.median(sil_clean)) if sil_clean.size else float('nan'),
            "silhouette_p25": float(np.percentile(sil_clean, 25)) if sil_clean.size else float('nan'),
            "silhouette_p75": float(np.percentile(sil_clean, 75)) if sil_clean.size else float('nan'),
        })

    # Save per-item silhouette if requested
    if SAVE_SILHOUETTE and np.isfinite(full_sil).any():
        per_item_path = OUTPUT_DIR / "per_item_silhouette.csv"
        with open(per_item_path, "w") as f:
            f.write("idx,class,silhouette\n")
            for i_local in range(F_bool.shape[0]):
                s = full_sil[i_local]
                if np.isnan(s):
                    continue
                cls = y_top[i_local]
                idx_global = int(idx_top[i_local])
                f.write(f"{idx_global},{cls},{float(s):.6f}\n")
        print(f"[INFO] Saved per-item silhouettes → {per_item_path}")

    # Save per-class compactness CSV (sorted by count desc)
    per_class.sort(key=lambda d: d["n"], reverse=True)
    per_class_path = OUTPUT_DIR / "per_class_compactness.csv"
    with open(per_class_path, "w") as f:
        f.write("class,n,mean_intra_similarity,median_intra_similarity,mean_similarity_to_medoid,mean_nearest_neighbor_similarity,silhouette_mean,silhouette_median,silhouette_p25,silhouette_p75\n")
        for d in per_class:
            f.write(
                f'{d["class"]},{d["n"]},{d["mean_intra_similarity"]:.6f},{d["median_intra_similarity"]:.6f},'
                f'{d["mean_similarity_to_medoid"]:.6f},{d["mean_nearest_neighbor_similarity"]:.6f},'
                f'{d["silhouette_mean"]:.6f},{d["silhouette_median"]:.6f},{d["silhouette_p25"]:.6f},{d["silhouette_p75"]:.6f}\n'
            )
    print(f"[INFO] Saved per-class compactness → {per_class_path}")

    # Inter-class: medoid-to-medoid Tanimoto matrix
    classes_final = [d["class"] for d in per_class]
    K = len(classes_final)
    M = np.eye(K, dtype=np.float32)
    # medoid rows in the subset matrix
    # Build a mapping from subset index to row in F_bool:
    #   medoids[cls] currently stores a SUBSET index (position in F_bool)
    for a, ca in enumerate(classes_final):
        ia = medoids[ca]
        Fa = F_bool[ia:ia+1]
        for b, cb in enumerate(classes_final):
            if b < a:
                M[a, b] = M[b, a]
                continue
            ib = medoids[cb]
            Fb = F_bool[ib:ib+1]
            d = pairwise_distances(Fa, Fb, metric='jaccard')[0, 0]
            M[a, b] = 1.0 - d
            M[b, a] = M[a, b]

    matrix_path = OUTPUT_DIR / "class_class_mean_tanimoto.csv"
    with open(matrix_path, "w") as f:
        f.write("," + ",".join(classes_final) + "\n")
        for i, ca in enumerate(classes_final):
            row = ",".join(f"{M[i, j]:.6f}" for j in range(K))
            f.write(f"{ca},{row}\n")
    print(f"[INFO] Saved class–class medoid Tanimoto matrix → {matrix_path}")

    # ---- PLOTS ----
    if K >= 2:
        plot_heatmap(M, classes_final, OUTPUT_DIR / "heatmap_class_class_tanimoto.png")
    if per_class:
        plot_bar_mean_intra(per_class, OUTPUT_DIR / "bar_mean_intra_similarity.png")
        plot_scatter_intra_vs_silhouette(per_class, OUTPUT_DIR / "scatter_intra_vs_silhouette.png")

if __name__ == "__main__":
    main()
