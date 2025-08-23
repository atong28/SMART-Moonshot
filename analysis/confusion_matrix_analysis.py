#!/usr/bin/env python3
"""
Top-K retrieval confusion by NP superclass.

For each test query:
  1) Build the query fingerprint (mirrors your ranking choice: binary / sigmoid / jaccard-binary).
  2) Retrieve Top-K items from the ranking corpus (cosine over ranker.data; already L2-normalized).
  3) Map retrieved indices -> NP superclass using ranking-set metadata.
  4) Tally counts: true superclass (row) vs retrieved superclass (col).

Outputs in OUTPUT_DIR:
  - topk_confusion_counts.csv        (raw counts)
  - topk_confusion_rownorm.csv       (row-normalized by (#queries * K))
  - topk_confusion_heatmap.png       (row-normalized heatmap)
  - top_confusions.txt               (sorted off-diagonal confusions)
"""

from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import json
import csv

# --- Your project imports ---
from archived_code.self_attention.src.settings import Args
from archived_code.self_attention.src.fp_loaders import get_fp_loader
from archived_code.self_attention.src.dataset import MoonshotDataModule, collate
from archived_code.self_attention.src.model import build_model

# ======================
# CONFIG
# ======================
CKPT = 'old_ckpt/trial_1.ckpt'   # edit if needed
DATA_ROOT = '/data/nas-gpu/wang/atong/MoonshotDataset'
INPUT_TYPES = ['hsqc', 'h_nmr', 'c_nmr', 'mw']

OUTPUT_DIR = Path('results/analysis/rank1_by_class_trial1_topk_5')  # reuse your folder
BATCH_SIZE = 16

# --- Confusion specifics ---
TOPK_CM = 5   # how many retrieved neighbors to count per query

# Path to ranking-set metadata aligned with model.ranker.data (len must match).
# Must contain, for each index, an object with key 'np_superclass' (list[str] or []).
RANKING_META_PATH = '/data/nas-gpu/wang/atong/MoonshotDataset/rankingset_meta.pkl'  # <-- set this!

# Whether to include query items that are Multi/Unknown in the confusion tallies.
INCLUDE_MULTI_QUERY = False
INCLUDE_UNKNOWN_QUERY = False

# If the retrieval corpus includes the query itself, drop Top-1 to avoid trivial self-match.
EXCLUDE_SELF_RETRIEVAL = False

# PNG settings
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
    # compatibility shim for older checkpoints
    if 'NMR_type_embedding.weight' in sd and 'embedding.weight' not in sd:
        sd['embedding.weight'] = sd['NMR_type_embedding.weight']; del sd['NMR_type_embedding.weight']
    model.load_state_dict(sd, strict=True)
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()
    return model, dm, device

def _load_rankingset_meta(path):
    if path is None:
        raise ValueError("RANKING_META_PATH is None. Please set it to the ranking-set metadata file.")
    if str(path).endswith(".pkl"):
        with open(path, "rb") as f:
            obj = pickle.load(f)
    elif str(path).endswith(".json"):
        with open(path, "r") as f:
            obj = json.load(f)
    else:
        raise ValueError(f"Unsupported ranking meta format: {path}")

    # Normalize to list style 0..N-1; each entry must have 'np_superclass'
    if isinstance(obj, list):
        meta = obj
    elif isinstance(obj, dict):
        try:
            N = len(obj)
            meta = [obj[str(i)] if str(i) in obj else obj[i] for i in range(N)]
        except Exception:
            raise ValueError("Ranking meta dict must be keyed by 0..N-1 (int or str).")
    else:
        raise ValueError("Ranking meta must be a list or dict.")
    for i, m in enumerate(meta):
        if not isinstance(m, dict) or 'np_superclass' not in m:
            raise ValueError(f"Meta item {i} missing 'np_superclass' key.")
    return meta

@torch.no_grad()
def _rank_queries_against_corpus(ranker, queries, topk: int) -> torch.Tensor:
    """
    ranker.data: (N, D) assumed L2-normalized by RankingSet
    queries: (B, D), will be L2-normalized here to match cosine
    returns LongTensor (B, topk) of corpus indices for each query
    """
    q = torch.nn.functional.normalize(queries, dim=1, p=2.0)
    sims = ranker.data @ q.T             # (N, B)
    idxs = torch.topk(sims, k=topk, dim=0).indices.T.contiguous()  # (B, topk)
    return idxs

def _first_superclass(label_list: List[str]) -> Tuple[str, str]:
    if not label_list:
        return 'unknown', None
    if len(label_list) == 1:
        return 'single', label_list[0]
    return 'multi', ';'.join(label_list)

def _pick_queries_for_ranking(model, logits, fp_label, use_jaccard: bool):
    """
    Mirror your cm(...) logic:
      - if use_jaccard: binary from logits>=0.0
      - elif model.rank_by_soft_output: sigmoid(logits)
      - else: binary from logits>=0.0
    """
    if use_jaccard:
        return (logits >= 0.0).float()
    if getattr(model, 'rank_by_soft_output', False):
        return torch.sigmoid(logits)
    return (logits >= 0.0).float()

def build_topk_confusion(
    model,
    dm,
    device,
    ranker,
    rankingset_meta: List[dict],
    topk: int = 10,
    exclude_self: bool = True,
    use_jaccard: bool = False,
    include_multi_query: bool = False,
    include_unknown_query: bool = False,
):
    """
    Returns:
      confusion_counts: Dict[true_cls -> Counter(pred_cls -> count)]
      class_counts: Counter(true_cls -> #queries considered)
    """
    confusion_counts: Dict[str, Counter] = defaultdict(Counter)
    class_counts = Counter()

    B = BATCH_SIZE
    N = len(dm.test)
    i = 0

    while i < N:
        j = min(i + B, N)
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
        queries = _pick_queries_for_ranking(model, logits, labels, use_jaccard=use_jaccard)

        # Get Top-K (+1 if we plan to drop the top self-match)
        tk = topk + (1 if exclude_self else 0)
        topk_idx = _rank_queries_against_corpus(ranker, queries, tk)  # (B, tk)

        # Tally per query
        for r, global_idx in enumerate(range(i, j)):
            # true class of the query item
            ltype, lval = get_np_superclass(dm, global_idx)
            if ltype == 'single':
                true_cls = lval
            elif ltype == 'multi':
                if not include_multi_query:
                    continue
                true_cls = 'Multi'
            else:
                if not include_unknown_query:
                    continue
                true_cls = 'Unknown'

            idxs = topk_idx[r].tolist()
            if exclude_self:
                # Pragmatic: drop Top-1 to avoid the query itself if corpus contains it.
                idxs = idxs[1:]
            idxs = idxs[:topk]

            for ridx in idxs:
                meta_labels = rankingset_meta[ridx].get('np_superclass', [])
                rtype, rval = _first_superclass(meta_labels)
                if rtype == 'single':
                    pred_cls = rval
                elif rtype == 'multi':
                    pred_cls = 'Multi'
                else:
                    pred_cls = 'Unknown'
                confusion_counts[true_cls][pred_cls] += 1

            class_counts[true_cls] += 1

        i = j

    return confusion_counts, class_counts

def _save_confusion_outputs(confusion_counts: Dict[str, Counter], class_counts: Counter, out_dir: Path, topk: int):
    out_counts = out_dir / "topk_confusion_counts.csv"
    out_norm   = out_dir / "topk_confusion_rownorm.csv"
    out_txt    = out_dir / "top_confusions.txt"
    out_png    = out_dir / "topk_confusion_heatmap.png"

    # assemble label ordering
    all_true = list(confusion_counts.keys())
    all_pred = set()
    for c in confusion_counts.values():
        all_pred.update(c.keys())
    pred_order = [p for p,_ in Counter({p: sum(c[p] for c in confusion_counts.values()) for p in all_pred}).most_common()]
    true_order = [t for t,_ in class_counts.most_common()]

    # counts CSV
    with open(out_counts, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["true_class"] + pred_order)
        for t in true_order:
            row = [t] + [confusion_counts[t][p] for p in pred_order]
            w.writerow(row)

    # row-normalized CSV (divide by (#queries * topk))
    with open(out_norm, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["true_class"] + pred_order)
        for t in true_order:
            denom = max(1, class_counts[t] * topk)
            row = [t] + [confusion_counts[t][p] / denom for p in pred_order]
            w.writerow(row)

    # top confusions (off-diagonal)
    pairs = []
    for t in true_order:
        for p in pred_order:
            if p != t:
                pairs.append(((t, p), confusion_counts[t][p]))
    pairs.sort(key=lambda x: x[1], reverse=True)
    with open(out_txt, "w") as f:
        for (t, p), c in pairs[:200]:
            f.write(f"{t} → {p}: {c}\n")

    # heatmap (row-normalized)
    M = np.array([[confusion_counts[t][p] for p in pred_order] for t in true_order], dtype=float)
    denom = np.array([max(1, class_counts[t] * topk) for t in true_order], dtype=float).reshape(-1, 1)
    M_norm = M / denom

    plt.figure(figsize=(max(8, 0.5*len(pred_order)), max(6, 0.4*len(true_order))))
    im = plt.imshow(M_norm, aspect='auto', interpolation='nearest')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(pred_order)), pred_order, rotation=60, ha='right', fontsize=8)
    plt.yticks(range(len(true_order)), true_order, fontsize=8)
    plt.xlabel("Retrieved superclass")
    plt.ylabel("True superclass")
    plt.title(f"Top-{topk} Retrieval Confusion (row-normalized)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=PNG_DPI, bbox_inches='tight'); plt.close()

    print(f"[INFO] Saved confusion counts CSV: {out_counts}")
    print(f".[INFO] Saved row-normalized CSV:  {out_norm}")
    print(f".[INFO] Saved heatmap PNG:          {out_png}")
    print(f".[INFO] Saved top confusions list:   {out_txt}")

# ======================
# Main
# ======================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model, dm, device = load_model_and_data()

    # Load and sanity-check rankingset metadata
    rankingset_meta = _load_rankingset_meta(RANKING_META_PATH)
    rs_N = len(rankingset_meta)
    if not hasattr(model, 'ranker') or model.ranker is None:
        raise RuntimeError("Model has no ranker; ensure model.setup_ranker() was called.")
    rs_mat = model.ranker.data
    if rs_mat is None:
        raise RuntimeError("Ranker has no 'data' buffer loaded.")
    if rs_N != rs_mat.size(0):
        raise ValueError(f"Ranking meta length ({rs_N}) != ranking set size ({rs_mat.size(0)}).")

    print("[INFO] Building Top-K retrieval confusion…")
    confusion_counts, class_counts = build_topk_confusion(
        model=model,
        dm=dm,
        device=device,
        ranker=model.ranker,
        rankingset_meta=rankingset_meta,
        topk=TOPK_CM,
        exclude_self=EXCLUDE_SELF_RETRIEVAL,
        use_jaccard=getattr(model, 'use_jaccard', False),
        include_multi_query=INCLUDE_MULTI_QUERY,
        include_unknown_query=INCLUDE_UNKNOWN_QUERY,
    )
    if not confusion_counts:
        print("[WARN] No queries satisfied the inclusion criteria.")
        return

    _save_confusion_outputs(confusion_counts, class_counts, OUTPUT_DIR, topk=TOPK_CM)

if __name__ == "__main__":
    main()
