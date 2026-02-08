import torch
import torch.nn as nn
import torch.nn.functional as F

class HSQCPeakEncoder(nn.Module):
    def __init__(self, d_model=256, mlp_hidden=256, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model),
        )
        self.query = nn.Parameter(torch.randn(d_model))  # [d_model]
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, peaks, peaks_mask):
        """
        peaks: [B, N, 3]
        peaks_mask: [B, N] True for real, False for pad
        returns:
          memory: [B, N, d_model]
          g: [B, d_model]
        """
        B, N, _ = peaks.shape
        h = self.mlp(peaks)              # [B, N, d]
        h = self.ln(h)

        # attention pooling: scores = q · h_i
        q = self.query.view(1, 1, -1)    # [1,1,d]
        scores = (h * q).sum(dim=-1)     # [B, N]
        scores = scores.masked_fill(~peaks_mask, -1e9)
        w = F.softmax(scores, dim=-1)    # [B, N]
        g = torch.bmm(w.unsqueeze(1), h).squeeze(1)  # [B, d]

        return self.dropout(h), self.dropout(g)
