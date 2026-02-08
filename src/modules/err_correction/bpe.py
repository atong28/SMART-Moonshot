import torch.nn as nn

from .encoder import HSQCPeakEncoder

class HSQC2Bag(nn.Module):
    def __init__(self, vocab_size, d_model=256, dropout=0.1):
        super().__init__()
        self.enc = HSQCPeakEncoder(d_model=d_model, dropout=dropout)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, peaks, peaks_mask):
        _, g = self.enc(peaks, peaks_mask)     # g: [B, d_model]
        logits = self.head(g)                  # [B, V]
        return logits
