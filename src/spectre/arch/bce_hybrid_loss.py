import torch
import torch.nn as nn
import torch.nn.functional as F

class BCECosineHybridLoss(nn.Module):
    """
    Hybrid loss for multilabel fingerprints:
      L = λ * BCEWithLogitsLoss + (1 - λ) * (1 - cosine(sigmoid(logits), targets))

    Args:
        lambda_bce (float): weight on BCE term (0..1). e.g., 0.6
        pos_weight (Tensor|float|None): same semantics as BCEWithLogitsLoss.
            - Per-bit: shape [D] for D=16384
            - Scalar: broadcast to all bits
        reduction (str): 'mean' (default) or 'sum' over the batch.
        cosine_eps (float): epsilon for cosine_similarity stability.
    """
    def __init__(self, lambda_bce: float = 0.6,
                 pos_weight: torch.Tensor | float | None = None,
                 reduction: str = "mean",
                 cosine_eps: float = 1e-8):
        super().__init__()
        assert 0.0 <= lambda_bce <= 1.0, "lambda_bce must be in [0,1]"
        self.lambda_bce = float(lambda_bce)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)
        self.reduction = reduction
        self.cosine_eps = cosine_eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, D) raw model outputs (pre-sigmoid)
        targets: (B, D) float in {0.,1.}
        """
        # BCE term (stable on logits)
        bce = self.bce(logits, targets)

        # Cosine term on probabilities for directional alignment
        probs = torch.sigmoid(logits)
        cos_per_sample = F.cosine_similarity(probs, targets, dim=-1, eps=self.cosine_eps)  # (B,)
        cos_loss = 1.0 - cos_per_sample  # (B,)

        if self.reduction == "mean":
            cos_loss = cos_loss.mean()
        elif self.reduction == "sum":
            cos_loss = cos_loss.sum()
        # else 'none' -> keep per-sample vector

        return self.lambda_bce * bce + (1.0 - self.lambda_bce) * cos_loss
