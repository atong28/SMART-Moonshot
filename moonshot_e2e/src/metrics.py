import torch
import torch.nn as nn
import numpy as np
from torchmetrics.classification import BinaryRecall, BinaryPrecision, BinaryF1Score, BinaryAccuracy
from .ranker import RankingSet

do_cos = nn.CosineSimilarity(dim=1)
def do_jaccard(pred, label):
    pred = pred > 0
    label = label > 0
    intersection = torch.sum(pred * label, axis=1)
    union = torch.sum((pred + label)>0, axis=1)
    return intersection / union
do_f1 = BinaryF1Score()
do_recall = BinaryRecall()
do_precision = BinaryPrecision()
do_accuracy = BinaryAccuracy()


def cm(model_output, fp_label, ranker: RankingSet, loss, loss_fn, thresh: float = 0.0, rank_by_soft_output=False, 
       query_idx_in_rankingset=None, mw=None,use_actual_mw_for_retrieval=None, use_jaccard = False, no_ranking=False):

    global do_f1, do_recall, do_precision, do_accuracy
    do_f1 = do_f1.to(model_output)
    do_recall = do_recall.to(model_output)
    do_precision = do_precision.to(model_output)
    do_accuracy = do_accuracy.to(model_output)

    # Fingeprint prediction
    fp_pred = (model_output >= thresh).float()

    # cos
    cos = torch.mean(do_cos(fp_label, fp_pred)).item()
    jaccard = torch.mean(do_jaccard(fp_label, fp_pred)).item()
    # bit activity
    active = torch.mean(torch.sum(fp_pred, axis=1)).item()
    # bit metrics
    f1 = do_f1(fp_pred, fp_label).item()
    prec = do_precision(fp_pred, fp_label).item()
    rec = do_recall(fp_pred, fp_label).item()
    acc = do_accuracy(fp_pred, fp_label).item()

    if np.isclose(thresh, 0.5):  # probabiltiies
        pos_contr = torch.where(
            fp_label == 0, torch.zeros_like(fp_label, dtype=torch.float), model_output)
        neg_contr = torch.where(
            fp_label == 1, torch.ones_like(fp_label, dtype=torch.float), model_output)
    elif np.isclose(thresh, 0.0):  # logits
        pos_contr = torch.where(
            fp_label == 0, -999 * torch.ones_like(fp_label, dtype=torch.float), model_output)
        neg_contr = torch.where(
            fp_label == 1, 999 * torch.ones_like(fp_label, dtype=torch.float), model_output)
    else:
        raise (f"Weird Threshold {thresh}")

    # print( pos_contr.device, fp_label.device)
    pos_loss = loss_fn(pos_contr, fp_label)
    neg_loss = loss_fn(neg_contr, fp_label)
    if no_ranking:
        return {
            f"ce_loss": loss.item(),
            f"pos_loss": pos_loss.item(),
            f"neg_loss": neg_loss.item(),
            f"pos_neg_loss": (pos_loss + neg_loss).item(),
            f"cos": cos,
            f"jaccard": jaccard,
            f"active_bits": active,
            f"f1": f1,
            f"precision": prec,
            f"recall": rec,
            f"accuracy": acc,
        }, None
    # === Do Ranking ===
    if use_jaccard:
        queries = fp_pred
    elif rank_by_soft_output:
        queries = torch.sigmoid(model_output)
    else:
        queries = fp_pred
    rank_res = ranker.batched_rank(queries, fp_label, query_idx_in_rankingset, mw=mw, use_actual_mw_for_retrieval=use_actual_mw_for_retrieval, use_jaccard=use_jaccard)
    cts = [1, 5, 10]
    # strictly less as batched_rank returns number of items STRICTLY greater
    ranks = {
        f"rank_{allow}": torch.sum(rank_res < allow).item() / torch.numel(rank_res)
        for allow in cts
    }
    # print(f"rank_1: {ranks['rank_1']}")
    # print("ranks", ranks)
    # exit(0)
    mean_rank = torch.mean(rank_res.float()).item()
    return {
        f"ce_loss": loss.item(),
        f"pos_loss": pos_loss.item(),
        f"neg_loss": neg_loss.item(),
        f"pos_neg_loss": (pos_loss + neg_loss).item(),
        f"cos": cos,
        f"jaccard": jaccard,
        f"active_bits": active,
        f"f1": f1,
        f"precision": prec,
        f"recall": rec,
        f"accuracy": acc,
        f"mean_rank": mean_rank,
        **ranks
    }, rank_res.view(-1)


