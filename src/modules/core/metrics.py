import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchmetrics.classification import (
    BinaryRecall,
    BinaryPrecision,
    BinaryF1Score,
    BinaryAccuracy,
)

from .ranker import RankingSet

# Cosine over bit vectors (row-wise)
do_cos = nn.CosineSimilarity(dim=1)


def do_jaccard(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """Binary Jaccard per row."""
    pred = pred > 0
    label = label > 0
    intersection = torch.sum(pred * label, dim=1)
    union = torch.sum((pred + label) > 0, dim=1).clamp_min(1)
    return intersection / union


# TorchMetrics (moved to correct device in cm())
do_f1 = BinaryF1Score()
do_recall = BinaryRecall()
do_precision = BinaryPrecision()
do_accuracy = BinaryAccuracy()


@torch.no_grad()
def cm(
    model_output: torch.Tensor,
    fp_label: torch.Tensor,
    ranker: RankingSet,
    loss: torch.Tensor,
    loss_fn,
    thresh: float = 0.0,
    no_ranking: bool = False,
):
    """
    Binary FP metrics + retrieval via cosine.

    Args:
        model_output (torch.Tensor): _description_
        fp_label (torch.Tensor): _description_
        ranker (RankingSet): _description_
        loss (torch.Tensor): _description_
        loss_fn (_type_): _description_
        thresh (float, optional): _description_. Defaults to 0.0.
        no_ranking (bool, optional): _description_. Defaults to False.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    global do_f1, do_recall, do_precision, do_accuracy

    # Ensure metrics are on the right device/dtype
    do_f1 = do_f1.to(model_output)
    do_recall = do_recall.to(model_output)
    do_precision = do_precision.to(model_output)
    do_accuracy = do_accuracy.to(model_output)

    # Binarize predictions at `thresh` for bitwise metrics
    fp_pred = (model_output >= thresh).float()

    # Vector-similarity diagnostics (using binarized predictions)
    cos = torch.mean(do_cos(fp_label, fp_pred)).item()
    jaccard = torch.mean(do_jaccard(fp_label, fp_pred)).item()

    # Bit activity (average number of 1s per row)
    active = torch.mean(torch.sum(fp_pred, dim=1)).item()

    # Bitwise classification metrics
    f1 = do_f1(fp_pred, fp_label).item()
    prec = do_precision(fp_pred, fp_label).item()
    rec = do_recall(fp_pred, fp_label).item()
    acc = do_accuracy(fp_pred, fp_label).item()

    # Positive/negative contribution losses (kept from original behavior)
    if np.isclose(thresh, 0.5):  # probabilities
        pos_contr = torch.where(fp_label == 0, torch.zeros_like(
            fp_label, dtype=torch.float), model_output)
        neg_contr = torch.where(fp_label == 1, torch.ones_like(
            fp_label, dtype=torch.float), model_output)
    elif np.isclose(thresh, 0.0):  # logits
        pos_contr = torch.where(
            fp_label == 0, -999 * torch.ones_like(fp_label, dtype=torch.float), model_output)
        neg_contr = torch.where(
            fp_label == 1,  999 * torch.ones_like(fp_label, dtype=torch.float), model_output)
    else:
        raise ValueError(f"Weird threshold {thresh}")

    pos_loss = loss_fn(pos_contr, fp_label)
    neg_loss = loss_fn(neg_contr, fp_label)

    base_metrics = {
        "loss": loss.item(),
        "pos_loss": pos_loss.item(),
        "neg_loss": neg_loss.item(),
        "pos_neg_loss": (pos_loss + neg_loss).item(),
        "cos": cos,
        "jaccard": jaccard,
        "active_bits": active,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "accuracy": acc,
    }
    if no_ranking:
        return base_metrics, None

    # Retrieval: use probabilities for queries; cosine mode
    queries = torch.sigmoid(model_output)
    rank_res = ranker.batched_rank(
        queries=queries,
        truths=fp_label,
        use_jaccard=False,
    )

    cts = [1, 5, 10]
    ranks = {f"rank_{k}": (rank_res < k).float().mean().item() for k in cts}
    mean_rank = rank_res.float().mean().item()

    return {
        **base_metrics,
        "mean_rank": mean_rank,
        **ranks,
    }, rank_res.view(-1)
