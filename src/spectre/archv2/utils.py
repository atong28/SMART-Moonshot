# SPECTREv2 — learned-first spectral encoders and model skeleton
# Assumptions: binary (entropy) fingerprints only.
#
# This single file is structured in sections you can split into modules:
# - utils: masks, small helpers
# - rbf: learned RBF banks (point and pairwise)
# - attention: multihead attention with additive bias (for Δm/z or δ-bias)
# - FiLM: lightweight MW conditioner
# - encoders.NMRSetEncoder: unified NMR set encoder (works with any subset of H/C/HSQC)
# - encoders.MSSetEncoder: MS/MS set encoder with learned Δm/z bias
# - model.SPECTREv2: end-to-end LightningModule using the above
#
# Integration points:
# - This fits the provided SPECTREDataModule (same batch dict keys, masks via all-zero rows).
# - Keep your existing metrics/loss imports.

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Optional, Tuple

# ========= utils ========= #

def seq_pad_mask(x: torch.Tensor) -> torch.Tensor:
    """Create a key_padding_mask for sequences padded with all-zero rows.
    x: (B, L, D)  →  mask: (B, L) with True = PAD
    """
    if x.numel() == 0:
        # (B, 0, D) safe path
        return torch.zeros((x.size(0), 0), dtype=torch.bool, device=x.device)
    return (x.abs().sum(dim=-1) == 0)

class IntensityGate(nn.Module):
    """Learned confidence gate from (log1p intensity) → multiplicative scale on features."""
    def __init__(self, d_model: int):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(1, d_model), nn.Sigmoid())
    def forward(self, feats: torch.Tensor, intensities: Optional[torch.Tensor]):
        if intensities is None:
            return feats
        gate = self.g(torch.log1p(torch.clamp(intensities, min=0.0)))  # (B,L,1)->(B,L,D)
        return feats * (0.5 + gate)

# ========= RBF banks ========= #

class RBFBank1D(nn.Module):
    """
    Learned RBF bank for a single axis (e.g., δH, δC, m/z).
    Maps (B,L,1) → (B,L,n_centers) → Linear → (B,L,d_out).
    """
    def __init__(self, n_centers: int, d_out: int, init_low: float, init_high: float):
        super().__init__()
        centers = torch.linspace(init_low, init_high, max(n_centers, 2)).view(1,1,-1)
        self.centers = nn.Parameter(centers)                 # learn centers (1,1,C)
        self.log_sigma = nn.Parameter(torch.zeros(centers.size(-1)))  # (C,)
        self.proj = nn.Linear(centers.size(-1), d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x.new_zeros(x.size(0), x.size(1), self.proj.out_features)
        r2 = (x - self.centers)**2                           # (B,L,C)
        sigma = torch.exp(self.log_sigma).clamp_min(1e-4).view(1,1,-1)
        phi = torch.exp(-r2 / (2.0 * sigma**2))              # (B,L,C)
        return self.proj(phi)

class RBFBankPairwise1D(nn.Module):
    """
    Learned RBF mixture over pairwise |Δx| to produce attention bias.
    Returns (B,Lq,Lk) scalar bias added to attention logits.
    """
    def __init__(self, n_centers: int, init_low: float, init_high: float, scale: float = 1.0):
        super().__init__()
        centers = torch.linspace(init_low, init_high, max(n_centers, 2))  # (C,)
        self.centers = nn.Parameter(centers)
        self.log_sigma = nn.Parameter(torch.zeros(centers.numel()))
        self.out_proj = nn.Linear(centers.numel(), 1, bias=False)
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, q: torch.Tensor, k: torch.Tensor, avail_q: Optional[torch.Tensor] = None, avail_k: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        q: (B,Lq,1), k: (B,Lk,1)
        avail_q: (B,Lq,1) 1 if coordinate present else 0; same for avail_k.
        """
        if q.size(1) == 0 or k.size(1) == 0:
            return q.new_zeros(q.size(0), q.size(1), k.size(1))
        d = (q - k.transpose(1,2)).abs().squeeze(-1)        # (B,Lq,Lk)
        centers = self.centers.view(1,1,1,-1)
        sigma = torch.exp(self.log_sigma).clamp_min(1e-4).view(1,1,1,-1)
        phi = torch.exp(-((d[...,None] - centers)**2) / (2.0 * sigma**2))  # (B,Lq,Lk,C)
        bias = self.out_proj(phi).squeeze(-1)               # (B,Lq,Lk)
        if (avail_q is not None) and (avail_k is not None):
            # zero out bias where either side lacks this coordinate
            mask = (avail_q > 0.5).float() @ (avail_k.transpose(1,2) > 0.5).float()
            bias = bias * mask
        return self.scale * bias

# ========= Attention with additive bias ========= #

class BiasMHA(nn.Module):
    """Multi-head attention with additive (B, Lq, Lk) bias on logits."""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                attn_bias: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, Lq, _ = q.shape
        def split(x):
            return x.view(B, -1, self.nhead, self.d_head).transpose(1, 2)  # (B,H,L,dh)

        Q = split(self.q_proj(q))
        K = split(self.k_proj(k))
        V = split(self.v_proj(v))

        logits = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)  # (B,H,Lq,Lk)
        if attn_bias is not None:
            logits = logits + attn_bias.unsqueeze(1)  # (B,H,Lq,Lk)

        mask_expanded = None
        if key_padding_mask is not None and key_padding_mask.numel() > 0:
            # (B,1,1,Lk) broadcastable to (B,H,Lq,Lk)
            mask_expanded = key_padding_mask.unsqueeze(1).unsqueeze(2)
            logits = logits.masked_fill(mask_expanded, -1e9)

        A = torch.softmax(logits, dim=-1)

        # ZERO rows where *every* key is masked (avoid residual leaking NaNs or noise)
        if mask_expanded is not None:
            # row_all_masked: (B,1,1,1) so we can expand to (B,H,Lq,1)
            row_all_masked = mask_expanded.all(dim=-1, keepdim=True)      # (B,1,1,1)
            A = A.masked_fill(row_all_masked.expand(-1, self.nhead, A.size(2), 1), 0.0)

        A = self.drop(A)
        out = torch.matmul(A, V)                                           # (B,H,Lq,dh)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)   # (B,Lq,D)
        return self.o_proj(out)

class TransformerBlockWithBias(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = BiasMHA(d_model, nhead, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor],
                key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        a = self.attn(x, x, x, attn_bias=attn_bias, key_padding_mask=key_padding_mask)
        x = self.ln1(x + self.drop(a))
        f = self.ff(x)
        x = self.ln2(x + self.drop(f))
        return x

# ========= FiLM (MW conditioner) ========= #

class FiLM(nn.Module):
    """Simple FiLM conditioner: MW scalar → (gamma, beta) per token feature."""
    def __init__(self, d_model: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * d_model)
        )
    def forward(self, mw: Optional[torch.Tensor], B: int, L: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if mw is None:
            gamma = torch.zeros(B, L,  self.net[-1].out_features // 2, device=device)
            beta  = torch.zeros(B, L,  self.net[-1].out_features // 2, device=device)
            return gamma, beta
        x = mw.view(B, 1, 1).expand(B, L, 1).contiguous()  # (B,L,1)
        gb = self.net(x)                                   # (B,L,2D)
        D = gb.size(-1) // 2
        gamma, beta = gb[..., :D], gb[..., D:]
        return gamma, beta

# ========= NMR unified encoder ========= #

class NMRSetEncoder(nn.Module):
    """
    Unifies HSQC (δH,δC), 1D 1H (δH only), 1D 13C (δC only) into a single token set.
    - Learned RBFs per axis (δH, δC) → concatenated + type embedding → proj to d_model
    - Learned pairwise bias from |ΔδH| and |ΔδC| via RBF mixtures
    - Optional FiLM conditioning by MW per layer

    Inputs (any subset may be missing):
      hsqc: (B, Lh2d, 2)  columns [δH, δC]
      h_nmr: (B, Lh, 3)   δH is column 1 (by your loader), others are zeros
      c_nmr: (B, Lc, 3)   δC is column 0 (by your loader), others are zeros
      mw: (B,) optional
    """
    def __init__(self, d_model: int, n_layers: int, nhead: int, d_ff: int,
                 dropout: float = 0.1,
                 n_rbf_H: int = 64, n_rbf_C: int = 64,
                 H_range: Tuple[float,float] = (0.0, 12.0),
                 C_range: Tuple[float,float] = (0.0, 220.0),
                 type_emb_dim: int = 32,
                 use_film: bool = True):
        super().__init__()
        self.d_model = d_model
        self.use_film = use_film
        # Axis encoders
        self.rbf_H = RBFBank1D(n_rbf_H, d_out=d_model//2, init_low=H_range[0], init_high=H_range[1])
        self.rbf_C = RBFBank1D(n_rbf_C, d_out=d_model//2, init_low=C_range[0], init_high=C_range[1])
        # Learned placeholders when axis missing
        self.missing_H = nn.Parameter(torch.zeros(1,1,d_model//2))
        self.missing_C = nn.Parameter(torch.zeros(1,1,d_model//2))
        # Type embeddings: 0=H_1D, 1=C_1D, 2=HSQC
        self.type_emb = nn.Embedding(3, type_emb_dim)
        self.in_proj = nn.Linear(d_model + type_emb_dim, d_model)
        # Pairwise bias from ΔδH and ΔδC (summed)
        self.bias_H = RBFBankPairwise1D(n_centers=32, init_low=0.0, init_high=2.0, scale=1.0)
        self.bias_C = RBFBankPairwise1D(n_centers=32, init_low=0.0, init_high=20.0, scale=1.0)
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlockWithBias(d_model, nhead, d_ff, dropout) for _ in range(n_layers)
        ])
        self.drop = nn.Dropout(dropout)
        if use_film:
            self.films = nn.ModuleList([FiLM(d_model) for _ in range(n_layers)])

    def _encode_axis(self, vals: Optional[torch.Tensor], rbf: RBFBank1D, missing_vec: nn.Parameter) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (features, availability) where availability is 1 if coord present else 0."""
        if vals is None or vals.size(1) == 0:
            return None, None
        # vals: (B,L,1)
        avail = (vals.abs() > 0).float()  # treat 0 as missing (fits your padded zeros)
        feats = rbf(vals) * avail + missing_vec.expand(vals.size(0), vals.size(1), -1) * (1.0 - avail)
        return feats, avail

    def forward(self, hsqc: Optional[torch.Tensor], h_nmr: Optional[torch.Tensor], c_nmr: Optional[torch.Tensor],
                mw: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # ---- Gather tokens ----
        B = 0
        device = None
        parts = []            # list of (feats, δH, δC, avail_H, avail_C, type_id)
        # HSQC: (B,L,2)
        if hsqc is not None:
            if hsqc.ndim == 2: hsqc = hsqc.unsqueeze(0)
            B = hsqc.size(0); device = hsqc.device
            H = hsqc[..., 0:1]
            C = hsqc[..., 1:2]
            fH, aH = self._encode_axis(H, self.rbf_H, self.missing_H)
            fC, aC = self._encode_axis(C, self.rbf_C, self.missing_C)
            feats = torch.cat([fH, fC], dim=-1)
            parts.append((feats, H, C, aH, aC, torch.full((B, hsqc.size(1), 1), 2, dtype=torch.long, device=hsqc.device)))
        # H 1D: δH at col 1
        if h_nmr is not None:
            if h_nmr.ndim == 2: h_nmr = h_nmr.unsqueeze(0)
            B = max(B, h_nmr.size(0)); device = h_nmr.device
            H = h_nmr[..., 1:2]
            fH, aH = self._encode_axis(H, self.rbf_H, self.missing_H)
            fC = self.missing_C.expand(B, h_nmr.size(1), -1)
            aC = torch.zeros(B, h_nmr.size(1), 1, device=h_nmr.device)
            feats = torch.cat([fH, fC], dim=-1)
            parts.append((feats, H, torch.zeros_like(H), aH, aC, torch.full((B, h_nmr.size(1), 1), 0, dtype=torch.long, device=h_nmr.device)))
        # C 1D: δC at col 0
        if c_nmr is not None:
            if c_nmr.ndim == 2: c_nmr = c_nmr.unsqueeze(0)
            B = max(B, c_nmr.size(0)); device = c_nmr.device
            C = c_nmr[..., 0:1]
            fC, aC = self._encode_axis(C, self.rbf_C, self.missing_C)
            fH = self.missing_H.expand(B, c_nmr.size(1), -1)
            aH = torch.zeros(B, c_nmr.size(1), 1, device=c_nmr.device)
            feats = torch.cat([fH, fC], dim=-1)
            parts.append((feats, torch.zeros_like(C), C, aH, aC, torch.full((B, c_nmr.size(1), 1), 1, dtype=torch.long, device=c_nmr.device)))

        if len(parts) == 0:
            # No NMR present
            empty = torch.zeros(B, 0, self.d_model, device=device)
            mask = torch.zeros(B, 0, dtype=torch.bool, device=device)
            H_all = torch.zeros(B, 0, 1, device=device); C_all = torch.zeros(B, 0, 1, device=device)
            aH_all = torch.zeros(B, 0, 1, device=device); aC_all = torch.zeros(B, 0, 1, device=device)
            return empty, mask, H_all, C_all

        feats = torch.cat([p[0] for p in parts], dim=1)
        H_all = torch.cat([p[1] for p in parts], dim=1)
        C_all = torch.cat([p[2] for p in parts], dim=1)
        aH_all = torch.cat([p[3] for p in parts], dim=1)
        aC_all = torch.cat([p[4] for p in parts], dim=1)
        types = torch.cat([p[5] for p in parts], dim=1).squeeze(-1)  # (B,L)
        type_feats = self.type_emb(types)                             # (B,L,Dt)
        x = self.in_proj(torch.cat([feats, type_feats], dim=-1))      # (B,L,D)
        x = self.drop(x)
        key_mask = (x.abs().sum(dim=-1) == 0)  # should be all False here, but keeps shape

        # Pairwise bias (sum of H and C)
        bH = self.bias_H(H_all, H_all, aH_all, aH_all)
        bC = self.bias_C(C_all, C_all, aC_all, aC_all)
        attn_bias = bH + bC

        # Transformer stack with optional FiLM(MW)
        for li, block in enumerate(self.blocks):
            if self.use_film:
                gamma, beta = self.films[li](mw, B=x.size(0), L=x.size(1), device=x.device)
                x = x * (1.0 + gamma) + beta
            x = block(x, attn_bias=attn_bias, key_padding_mask=key_mask)
        return x, key_mask, H_all, C_all

# ========= MS/MS encoder ========= #

class MSSetEncoder(nn.Module):
    """
    MS/MS set encoder with learned RBF features on m/z and learned Δm/z bias in attention.
    Inputs:
      mass_spec: (B,L,3) where col0=m/z, col1=intensity, col2=padding zero (per your loader)
      mw: (B,) optional (for FiLM)
    """
    def __init__(self, d_model: int, n_layers: int, nhead: int, d_ff: int,
                 dropout: float = 0.1,
                 n_rbf_mz: int = 64, mz_range: Tuple[float,float] = (10.0, 2000.0),
                 n_rbf_d: int = 64, d_range: Tuple[float,float] = (0.0, 300.0),
                 use_isotope_bias: bool = True,
                 use_film: bool = True):
        super().__init__()
        self.d_model = d_model
        self.use_film = use_film
        self.use_isotope_bias = use_isotope_bias
        self.rbf_mz = RBFBank1D(n_rbf_mz, d_out=d_model, init_low=mz_range[0], init_high=mz_range[1])
        self.int_gate = IntensityGate(d_model)
        # Pairwise Δm/z bias
        self.bias_d = RBFBankPairwise1D(n_centers=n_rbf_d, init_low=d_range[0], init_high=d_range[1], scale=1.0)
        if use_isotope_bias:
            # initialize broadly around small gaps; learned thereafter
            self.bias_iso = RBFBankPairwise1D(n_centers=8, init_low=0.5, init_high=2.5, scale=0.5)
        self.blocks = nn.ModuleList([
            TransformerBlockWithBias(d_model, nhead, d_ff, dropout) for _ in range(n_layers)
        ])
        self.drop = nn.Dropout(dropout)
        if use_film:
            self.films = nn.ModuleList([FiLM(d_model) for _ in range(n_layers)])

    def forward(self, mass_spec: Optional[torch.Tensor], mw: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if mass_spec is None:
            device = mw.device if mw is not None else torch.device('cpu')
            return torch.zeros(0, 0, self.d_model, device=device), torch.zeros(0, 0, dtype=torch.bool, device=device)
        if mass_spec.ndim == 2:
            mass_spec = mass_spec.unsqueeze(0)

        B, L, D = mass_spec.shape
        mz   = mass_spec[..., 0:1]
        inten= mass_spec[..., 1:2]

        # Base features + intensity gate
        x = self.rbf_mz(mz)
        x = self.int_gate(x, inten)
        x = self.drop(x)

        key_mask = seq_pad_mask(mass_spec)           # (B,L) True=PAD
        all_pad  = key_mask.all(dim=1)               # (B,) True if this sample has NO real MS key

        if all_pad.any():
            x[all_pad] = 0.0

        # Pairwise bias
        b = self.bias_d(mz, mz)
        if self.use_isotope_bias:
            b = b + self.bias_iso(mz, mz)

        # Stack
        for li, block in enumerate(self.blocks):
            if self.use_film:
                gamma, beta = self.films[li](mw, B=B, L=L, device=mass_spec.device)
                # NEW: avoid injecting MW into missing-MS samples
                if all_pad.any():
                    gamma[all_pad] = 0.0
                    beta[all_pad]  = 0.0
                x = x * (1.0 + gamma) + beta
            x = block(x, attn_bias=b, key_padding_mask=key_mask)
            # NEW: keep missing-MS rows as zeros after each block (residual safety)
            if all_pad.any():
                x[all_pad] = 0.0
        return x, key_mask


# ========= Cross-attention block (CLS → joint) ========= #

class CrossAttentionBlock(nn.Module):
    """Query (global CLS) attends to Key/Value (the spectral tokens)."""
    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = BiasMHA(d_model, nhead, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        a = self.attn(query, key, value, attn_bias=None, key_padding_mask=key_padding_mask)
        x = self.ln1(query + self.drop(a))
        f = self.ff(x)
        x = self.ln2(x + self.drop(f))
        return x
