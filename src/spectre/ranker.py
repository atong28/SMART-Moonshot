import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .utils import set_float32_highest_precision


@set_float32_highest_precision
class RankingSet(torch.nn.Module):
    """
    Minimal fast similarity-ranking over a bank of Morgan fingerprints.

    Stores:
      - `data`: (N, D) float32 fingerprint matrix (NOT normalized here).

    Typical usage:
      >>> ranker = RankingSet(store=fps)  # fps: (N, D) float32 tensor
      >>> idxs = ranker.retrieve_idx(query_fp, n=50)
      >>> counts = ranker.batched_rank(queries, truths, use_jaccard=False)
    """

    def __init__(self, store: torch.Tensor, debug: bool = False):
        """
        Args:
            store: (N, D) float32 fingerprint matrix to rank against.
            debug: if True, logs extra info during ranking.
        """
        super().__init__()
        self.logger = logging.getLogger("lightning")
        self.logger.setLevel(logging.DEBUG)

        self.debug = debug
        if store.dtype != torch.float32:
            store = store.to(torch.float32)

        # Keep as a buffer so it moves with .to(device) but isn't a parameter
        self.register_buffer("data", store, persistent=False)
        self.logger.info(f"[RankingSet] Initialized with {len(self.data)} sample(s)")

    @property
    def device(self) -> torch.device:
        return self.data.device

    # -------- Utilities --------

    @staticmethod
    def round(fp: torch.Tensor) -> torch.Tensor:
        """
        Convert a fingerprint vector to a binary vector with 1s at indices equal to the max value.

        Args:
            fp: (D,) or (1, D)

        Returns:
            (D,) binary tensor.
        """
        fp = fp.flatten()
        hi = torch.max(fp)
        out = torch.zeros_like(fp)
        out[fp == hi] = 1
        return out

    @staticmethod
    def normalized_to_nonzero(fp: torch.Tensor) -> Tuple[int, ...]:
        """
        Indices whose values equal (within tolerance) the max of `fp`.
        """
        hi = torch.max(fp)
        nonzero = torch.nonzero(torch.isclose(fp, hi), as_tuple=False)
        return tuple(nonzero[:, 0].tolist())

    # -------- Retrieval & ranking --------

    def retrieve_idx(self, query: torch.Tensor, n: int = 50) -> torch.Tensor:
        """
        Top-N nearest neighbors by cosine similarity.

        Args:
            query: (D,) or (B, D); normalized internally.
            n: number of neighbors.

        Returns:
            (n,) if single query; else (n, B) indices.
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)
        query = F.normalize(query.to(self.device), dim=1, p=2.0)  # (B, D)
        sims = self.data @ query.T                                 # (N, B)
        _, idxs = torch.topk(sims, k=min(n, sims.size(0)), dim=0)
        return idxs.squeeze(1) if idxs.size(1) == 1 else idxs

    def jaccard_rank(
        self,
        data: torch.Tensor,
        queries: torch.Tensor,
        truths: torch.Tensor,
        thresh: torch.Tensor,
        query_idx_in_rankingset: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Count, per query, how many entries in `data` meet/exceed the Jaccard threshold.

        Args:
            data    : (N, D) binary/non-negative tensor.
            queries : (Q, D) binary tensor.
            truths  : (Q, D) binary tensor (shape check only).
            thresh  : (Q,) per-query Jaccard thresholds.

        Returns:
            (Q,) int32 counts, minus 1 to ignore the query's own row if present.
        """
        assert queries.size() == truths.size(), "queries and truths must share shape"
        counts = []
        with torch.no_grad():
            for i, q in enumerate(queries):
                inter = torch.sum((data * q) > 0, dim=1)
                union = torch.sum((data + q) > 0, dim=1).clamp_min(1)
                jacc = inter / union
                counts.append(torch.sum(jacc >= thresh[i], dtype=torch.int32))
        out = torch.stack(counts).to(torch.int32)
        return out - 1  # discount the gold row if present

    def dot_prod_rank(
        self,
        data: torch.Tensor,
        queries: torch.Tensor,
        truths: torch.Tensor,
        thresh: torch.Tensor,
        query_idx_in_rankingset: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Count, per query, how many entries in `data` meet/exceed the cosine-sim threshold.

        Expects `queries` and `truths` already L2-normalized.
        Returns (Q,) int32 counts, minus 1 to ignore the self-row.
        """
        assert queries.size() == truths.size(), "queries and truths must share shape"
        with torch.no_grad():
            sims = data @ queries.T  # (N, Q)
            ct = torch.sum(
                torch.logical_or(sims >= thresh, torch.isclose(sims, thresh)),
                dim=0,
                keepdim=True,
                dtype=torch.int32,
            )
            ct = ct - 1
            if self.debug:
                truth_sims = data @ truths.T
                self.logger.debug(f"truth_sims shape: {truth_sims.shape}")
                self.logger.debug(f"ct_greater:\n{ct}")
            return ct.squeeze(0)

    def batched_rank(
        self,
        queries: torch.Tensor,
        truths: torch.Tensor,
        query_idx_in_rankingset: Optional[torch.Tensor] = None,
        use_jaccard: bool = False,
    ) -> torch.Tensor:
        """
        Rank each query against `self.data` and count how many entries beat its query-specific threshold.

        If `use_jaccard`:
            - Threshold = Jaccard(query, truth) per query (binary).
        Else:
            - Threshold = cosine(query_i, truth_i) per query (after L2-normalization).

        Args:
            queries: (Q, D)
            truths : (Q, D)

        Returns:
            (Q,) int32 counts (minus 1 for the self-row if present).
        """
        with torch.no_grad():
            if use_jaccard:
                intersection = torch.sum(queries * truths, dim=1)
                union = torch.sum((queries + truths) > 0, dim=1).clamp_min(1)
                thresh = intersection / union  # (Q,)
                return self.jaccard_rank(self.data, queries, truths, thresh, query_idx_in_rankingset)
            else:
                qn = F.normalize(queries, dim=1, p=2.0)
                tn = F.normalize(truths, dim=1, p=2.0)
                thresh = torch.sum((qn * tn).T, dim=0, keepdim=True)  # (1, Q)
                return self.dot_prod_rank(self.data, qn, tn, thresh, query_idx_in_rankingset)
