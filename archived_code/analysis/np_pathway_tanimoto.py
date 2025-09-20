# np_pathway_umap_points_single.py
# Top-20 NP pathways — UMAP with tiny points:
#   (A) CLS embeddings (model Trial 1)
#   (B) Morgan 2048 fingerprints with Tanimoto/Jaccard

from pathlib import Path
from collections import Counter
import numpy as np
import torch
import matplotlib.pyplot as plt

# ---- Project imports (your repo) ----
from archived_code.self_attention.src.settings import Args
from archived_code.self_attention.src.fp_loaders import get_fp_loader
from archived_code.self_attention.src.dataset import MoonshotDataModule, collate
from archived_code.self_attention.src.model import build_model

# ---- External deps ----
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples  # optional (not required to run)

# RDKit for Morgan fingerprints
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
from matplotlib.patches import Patch

# ======================
# CONFIG — tweak here
# ======================
CKPT = 'old_ckpt/trial_1.ckpt'      # single model (Trial 1)
DATA_ROOT = '/data/nas-gpu/wang/atong/MoonshotDataset'
INPUT_TYPES = ['hsqc', 'h_nmr', 'c_nmr', 'mw']

OUTPUT_DIR = Path('results/analysis/np_pathway_top20_trial1_points')
TOP_K_CLASSES = 20
BATCH_SIZE = 64

# UMAP params
UMAP_NEIGHBORS = 20
UMAP_MIN_DIST = 0.08
UMAP_SEED = 42

# Scatter appearance
POINT_SIZE = 10.0       # tiny, crisp dots
POINT_ALPHA = 1.0
PNG_DPI = 400
MAX_PER_CLASS = None   # e.g., 5000 to cap points per class

# ======================
# Helpers
# ======================
def get_np_pathway_label(dm, idx):
    labels = dm.test.data[idx][1].get('np_pathway', [])
    if not labels: return 'unknown', None
    if len(labels) == 1: return 'single', labels[0]
    return 'multi', ';'.join(labels)

def select_top_k_classes(y_single, k=20):
    counts = Counter(y_single)
    items = [(c, n) for c, n in counts.items() if n >= 2]
    items.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in items[:k]]

def load_model_from_ckpt(args, fp_loader, dm, ckpt_path, device=None):
    model = build_model(args, True, fp_loader, dm.combinations_names)
    sd = torch.load(ckpt_path, weights_only=True)['state_dict']
    # key rename per your earlier code
    sd['embedding.weight'] = sd['NMR_type_embedding.weight']; del sd['NMR_type_embedding.weight']
    model.load_state_dict(sd, strict=True)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    return model, device

def collect_test_cls_embeddings(model, dm, batch_size=64, device=None):
    model.eval()
    device = device or next(model.parameters()).device
    cls_all, y_types_vals = [], []
    with torch.no_grad():
        N = len(dm.test)
        i = 0
        while i < N:
            j = min(i + batch_size, N)
            batch = [dm.test[k] for k in range(i, j)]
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
                y_types_vals.append(get_np_pathway_label(dm, k))
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

def plot_points_umap_single(
    Z, labels, classes, out_png=None, out_pdf=None,
    point_size=2.2, point_alpha=0.75, max_per_class=None,
    palette="tab20", shuffle=True, rasterized_pdf=True, dpi=400,
    title=None
):
    cmap = plt.get_cmap(palette)
    colors = [np.array(cmap(i)[:3], dtype=np.float32) for i in range(len(classes))]

    # per-class indices (shuffle + cap)
    rng = np.random.RandomState(0)
    idx_by_class = []
    for cls in classes:
        idx = np.where(labels == cls)[0]
        if shuffle: rng.shuffle(idx)
        if (max_per_class is not None) and (idx.size > max_per_class):
            idx = idx[:max_per_class]
        idx_by_class.append(idx)

    # round-robin interleave so no class hogs the top layer
    interleaved = []
    k = 0
    while True:
        added = False
        for j, ix in enumerate(idx_by_class):
            if k < len(ix):
                interleaved.append((j, ix[k]))
                added = True
        if not added: break
        k += 1

    fig, ax = plt.subplots(figsize=(8.8, 7.6))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title or "Top-20 NP pathways — UMAP points", fontsize=12, pad=10)

    for j, cls in enumerate(classes):
        pts = np.array([Z[i] for c,i in interleaved if c == j])
        if pts.size == 0: continue
        ax.scatter(
            pts[:,0], pts[:,1],
            s=point_size, alpha=point_alpha,
            c=[colors[j]], edgecolors="none",
            linewidths=0, rasterized=rasterized_pdf
        )

    handles = [Patch(facecolor=colors[j], edgecolor='none', label=str(cls)) for j, cls in enumerate(classes)]
    ncol = 2 if max(len(str(c)) for c in classes) > 14 else 1
    ax.legend(handles=handles, loc='upper right', fontsize=8, frameon=True, ncol=ncol)
    plt.tight_layout()
    if out_png: fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    if out_pdf: fig.savefig(out_pdf, bbox_inches="tight")
    return fig

def smiles_list_for_test(dm):
    return [dm.test.data[i][1].get('smiles', '') for i in range(len(dm.test))]

def morgan_fp_matrix(smiles_list, idx_subset, n_bits=2048, radius=2):
    """Return (F, ok_mask) where F is (M, n_bits) uint8 and ok_mask marks valid SMILES rows."""
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
    return F, ok

# ======================
# Main
# ======================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Data
    args = Args(**{'data_root': DATA_ROOT, 'input_types': INPUT_TYPES})
    fp_loader = get_fp_loader(args)
    dm = MoonshotDataModule(args, str(OUTPUT_DIR), fp_loader)
    dm.setup('fit'); dm.setup('test')

    # Model
    model, device = load_model_from_ckpt(args, fp_loader, dm, CKPT)

    # ---- CLS embeddings → UMAP points ----
    X_all, idx_single, y_single = collect_test_cls_embeddings(model, dm, batch_size=BATCH_SIZE, device=device)
    top20 = select_top_k_classes(y_single, k=TOP_K_CLASSES)
    print(f"[INFO] Top-{len(top20)} classes:", top20)

    m_top = np.isin(y_single, top20)
    X_cls = X_all[idx_single[m_top]]
    y_top = y_single[m_top]

    cls_reducer = SharedUMAP(n_neighbors=UMAP_NEIGHBORS, min_dist=UMAP_MIN_DIST, random_state=UMAP_SEED).fit(X_cls)
    Z_cls = cls_reducer.transform(X_cls)

    png_cls = OUTPUT_DIR / 'top20_umap_points_cls_trial1.png'
    pdf_cls = OUTPUT_DIR / 'top20_umap_points_cls_trial1.pdf'
    plot_points_umap_single(
        Z_cls, y_top, top20,
        out_png=png_cls, out_pdf=pdf_cls,
        point_size=POINT_SIZE, point_alpha=POINT_ALPHA, max_per_class=MAX_PER_CLASS,
        palette="tab20", shuffle=True, rasterized_pdf=True, dpi=PNG_DPI,
        title="Top-20 NP pathways — UMAP points (CLS, Trial 1)"
    )
    print(f"[INFO] Saved CLS UMAP → {png_cls}")
    print(f"[INFO] Saved CLS UMAP → {pdf_cls}")

    # ---- Morgan 2048 (Tanimoto/Jaccard) → UMAP points ----
    print("[INFO] Computing Morgan 2048 fingerprints for Tanimoto UMAP...")
    smiles = smiles_list_for_test(dm)
    idx_top = idx_single[m_top]         # indices into test set aligning y_top
    F, ok = morgan_fp_matrix(smiles, idx_top, n_bits=2048, radius=2)
    if not np.all(ok):
        F = F[ok]
        y_top = y_top[ok]

    tani_reducer = umap.UMAP(
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="jaccard",              # Tanimoto for binary vectors
        random_state=UMAP_SEED
    ).fit(F)
    Z_tani = tani_reducer.embedding_

    png_tani = OUTPUT_DIR / 'top20_umap_points_tanimoto_trial1.png'
    pdf_tani = OUTPUT_DIR / 'top20_umap_points_tanimoto_trial1.pdf'
    plot_points_umap_single(
        Z_tani, y_top, top20,
        out_png=png_tani, out_pdf=pdf_tani,
        point_size=POINT_SIZE, point_alpha=POINT_ALPHA, max_per_class=MAX_PER_CLASS,
        palette="tab20", shuffle=True, rasterized_pdf=True, dpi=PNG_DPI,
        title="Top-20 NP pathways — UMAP points (Morgan 2048, Tanimoto)"
    )
    print(f"[INFO] Saved Tanimoto UMAP → {png_tani}")
    print(f"[INFO] Saved Tanimoto UMAP → {pdf_tani}")

if __name__ == "__main__":
    main()
