from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import pytorch_lightning as pl
from ..core.settings import SPECTREArgs
from ..arch.bce_hybrid_loss import BCECosineHybridLoss
from ..core.metrics import cm
from ..core.const import ELEM2IDX
from ..core.ranker import RankingSet
from ..data.fp_loader import FPLoader
from .utils import NMRSetEncoder, MSSetEncoder, CrossAttentionBlock

class SPECTREv2(pl.LightningModule):
    def __init__(self, args: SPECTREArgs, fp_loader: FPLoader, use_film_mw: bool = True):
        super().__init__()
        self.args = args
        self.out_dim = args.out_dim
        self.dim_model = args.dim_model
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.loss = BCECosineHybridLoss(lambda_bce=args.lambda_hybrid)  # uses BCE + cosine
        # Encoders
        self.enc_nmr = NMRSetEncoder(args.dim_model, args.self_attn_layers, args.heads, args.ff_dim, dropout=args.dropout, use_film=use_film_mw)
        self.enc_ms  = MSSetEncoder(args.dim_model, args.self_attn_layers, args.heads, args.ff_dim, dropout=args.dropout, use_film=use_film_mw)
        # Formula tokens (same idea as v1)
        num_elem_tokens = len(ELEM2IDX) + 1  # +1 for PAD=0
        self.elem_embed = nn.Embedding(num_elem_tokens, args.dim_model, padding_idx=0)
        self.cnt_embed  = nn.Linear(1, args.dim_model)
        # MW token (keep explicit token in addition to FiLM)
        self.mw_embed = nn.Linear(1, args.dim_model)
        # Global CLS & cross-attention
        self.global_cls = nn.Parameter(torch.randn(1, 1, args.dim_model))
        self.cross_blocks = nn.ModuleList([
            CrossAttentionBlock(args.dim_model, args.heads, args.ff_dim, args.dropout) for _ in range(args.layers)
        ])
        self.fc = nn.Linear(args.dim_model, args.out_dim)
        self.fp_loader = fp_loader

    # ---- helpers ---- #
    @staticmethod
    def _maybe(x: Dict[str, torch.Tensor], key: str) -> Optional[torch.Tensor]:
        return x.get(key, None)

    def _formula_tokens(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if ('elem_idx' not in batch) or ('elem_cnt' not in batch):
            B = next(iter(batch.values())).size(0)
            return torch.zeros(B, 0, self.dim_model, device=self.device), torch.zeros(B, 0, dtype=torch.bool, device=self.device)
        eidx = batch['elem_idx']                      # (B, L)
        ecnt = batch['elem_cnt'].unsqueeze(-1).float()# (B, L, 1)
        ve = self.elem_embed(eidx)                    # (B, L, D)
        vc = self.cnt_embed(ecnt)                     # (B, L, D)
        tok = ve + vc
        mask = (eidx == 0)
        return tok, mask

    # ---- forward ---- #
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        B = next(iter(batch.values())).size(0)
        mw = batch.get('mw', None)
        if mw is not None and mw.ndim == 1:
            mw = mw.to(self.device)
        # NMR
        nmr_tokens, nmr_mask, _, _ = self.enc_nmr(
            self._maybe(batch, 'hsqc'), self._maybe(batch, 'h_nmr'), self._maybe(batch, 'c_nmr'), mw
        )
        # MS
        ms_tokens, ms_mask = self.enc_ms(self._maybe(batch, 'mass_spec'), mw)
        # MW token
        mw_tok = None
        if mw is not None:
            mw_tok = self.mw_embed(mw.view(B,1,1)).view(B,1,-1)
        # Formula tokens
        frm_tokens, frm_mask = self._formula_tokens(batch)
        # Joint seq + mask
        seqs = [t for t in [nmr_tokens, ms_tokens, frm_tokens, mw_tok] if t is not None and t.size(1) > 0]
        masks= [m for m in [nmr_mask,  ms_mask,  frm_mask,  None]      if (m is None) or (m.numel() > 0)]
        if len(seqs) == 0:
            joint = torch.zeros(B, 0, self.dim_model, device=self.device)
            joint_mask = torch.zeros(B, 0, dtype=torch.bool, device=self.device)
        else:
            joint = torch.cat(seqs, dim=1)
            # build mask; MW token has no padding
            pad_masks = []
            for m in masks:
                if m is None:  # MW token
                    pad_masks.append(torch.zeros(B, 1, dtype=torch.bool, device=joint.device))
                else:
                    pad_masks.append(m)
            joint_mask = torch.cat(pad_masks, dim=1)
        # Cross-attend from global CLS
        q = self.global_cls.expand(B, 1, -1)
        for blk in self.cross_blocks:
            q = blk(q, key=joint, value=joint, key_padding_mask=joint_mask)
        logits = self.fc(q.squeeze(1))  # (B, out_dim); keep as logits for BCE hybrid loss
        return logits

    def training_step(self, batch, batch_idx):
        batch_inputs, fps = batch
        logits = self.forward(batch_inputs)
        loss = self.loss(logits, fps)
        self.log('tr/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_inputs, fps = batch
        logits = self.forward(batch_inputs)
        loss = self.loss(logits, fps)
        metrics, _ = cm(logits, fps, ranker=None, loss=loss, loss_fn=self.loss,
                        thresh=0.0, query_idx_in_rankingset=batch_idx, no_ranking=True)
        for metric, value in metrics.items():
            self.log(f'val/mean_{metric}', value, on_epoch=True, prog_bar=False, sync_dist=True)
        return metrics

    def test_step(self, batch, batch_idx):
        batch_inputs, fps = batch
        logits = self.forward(batch_inputs)
        loss = self.loss(logits, fps)
        metrics, _ = cm(logits, fps, ranker=self.ranker, loss=loss, loss_fn=self.loss,
                        thresh=0.0, query_idx_in_rankingset=None, no_ranking=False)
        for metric, value in metrics.items():
            self.log(f'test/mean_{metric}', value, on_epoch=True, prog_bar=False, sync_dist=True)
        return metrics

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return opt

    def setup_ranker(self):
        assert self.args.fp_type == 'RankingEntropy', 'Only RankingEntropy fp type supported'
        store = self.fp_loader.load_rankingset(self.args.fp_type)
        metric = "cosine"
        self.ranker = RankingSet(store=store, metric=metric)