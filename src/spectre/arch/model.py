from typing import List, Dict, Any
import logging
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.distributed as dist
from collections import defaultdict
import numpy as np

from ..core.const import ELEM2IDX
from ..core.settings import SPECTREArgs
from ..core.utils import L1
from ..core.metrics import cm
from ..core.ranker import RankingSet
from ..core.lr_scheduler import NoamOpt
from ..data.fp_loader import FPLoader
from ..data.encoder import build_encoder
from .attention import MultiHeadAttentionCore
from .bce_hybrid_loss import BCECosineHybridLoss

logger = logging.getLogger("lightning")
if dist.is_initialized():
    rank = dist.get_rank()
    if rank != 0:
        logger.setLevel(logging.WARNING)
logger_should_sync_dist = torch.cuda.device_count() > 1

class CrossAttentionBlock(nn.Module):
    """
    Query (global CLS) attends to Key/Value (the spectral peaks + other tokens).
    """
    def __init__(self, dim_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttentionCore(
            embed_dim=dim_model,
            num_heads=num_heads,
            dropout=dropout,
            bias=True,
        )
        self.norm1 = nn.LayerNorm(dim_model)
        self.ff = nn.Sequential(
            nn.Linear(dim_model, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim_model),
        )
        self.norm2 = nn.LayerNorm(dim_model)

    def forward(self, query, key, value, key_padding_mask=None):
        attn_out = self.attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
        )
        q1 = self.norm1(query + attn_out)
        ff_out = self.ff(q1)
        out = self.norm2(q1 + ff_out)
        return out

class SPECTRE(pl.LightningModule):
    def __init__(self, args: SPECTREArgs, fp_loader: FPLoader):
        super().__init__()
        
        self.args = args
        self.fp_loader = fp_loader
        
        if self.global_rank == 0:
            logger.info("[SPECTRE] Started Initializing")

        self.fp_length = args.out_dim
        self.out_dim = args.out_dim
        
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.noam_factor = args.noam_factor
        self.weight_decay = args.weight_decay
        self.heads = args.heads
        self.layers = args.layers
        self.ff_dim = args.ff_dim
        self.dropout = args.dropout
        self.l1_decay = args.l1_decay

        self.scheduler = args.scheduler
        self.warm_up_steps = args.warm_up_steps
        self.dim_model = args.dim_model
        
        self.use_jaccard = args.use_jaccard
        
        self.freeze_weights = args.freeze_weights

        # ranked encoder
        self.enc_nmr = build_encoder(
            args.dim_model,
            args.nmr_dim_coords,
            [args.c_wavelength_bounds, args.h_wavelength_bounds],
            args.use_peak_values
        )
        self.enc_ms = build_encoder(
            args.dim_model,
            args.ms_dim_coords,
            [args.mz_wavelength_bounds, args.intensity_wavelength_bounds],
            args.use_peak_values
        )

        # 1) coordinate encoders
        self.encoders = {
            "hsqc": self.enc_nmr,
            "h_nmr": self.enc_nmr,
            "c_nmr": self.enc_nmr,
            "mass_spec": self.enc_ms
        }
        self.encoders = nn.ModuleDict({k: v for k, v in self.encoders.items() if k in self.args.input_types})
        
        self.self_attn = nn.ModuleDict({
            modality: nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.dim_model, nhead=self.heads,
                    dim_feedforward=self.ff_dim,
                    batch_first=True, dropout=self.dropout
                ), 
                num_layers= args.self_attn_layers
            )
            for modality in self.encoders
        })
        
        self.cross_blocks = nn.ModuleList([
            CrossAttentionBlock(
                dim_model=self.dim_model,
                num_heads=self.heads,
                ff_dim=self.ff_dim,
                dropout=self.dropout
            )
            for _ in range(self.layers)
        ])
        
        self.global_cls = nn.Parameter(torch.randn(1, 1, self.dim_model))
        self.mw_embed = nn.Linear(1, self.dim_model)
        self.fc = nn.Linear(self.dim_model, self.out_dim)
        num_elem_tokens = len(ELEM2IDX) + 1    # +1 for PAD=0
        self.elem_embed = nn.Embedding(
            num_embeddings = num_elem_tokens,
            embedding_dim  = self.dim_model,
            padding_idx    = 0
        )
        self.cnt_embed = nn.Linear(1, self.dim_model)

        if args.hybrid_loss:
            self.loss = BCECosineHybridLoss(lambda_bce = args.lambda_bce)
        else:
            self.loss = nn.BCEWithLogitsLoss()

        self.validation_step_outputs: List[Dict[str, Any]] = []
        self.test_step_outputs: List[Dict[str, Any]] = []
        
        if self.freeze_weights:
            for parameter in self.parameters():
                parameter.requires_grad = False
        
        if self.l1_decay > 0:
            self.cross_attn = L1(self.cross_attn, self.l1_decay)
            self.self_attn  = L1(self.self_attn,  self.l1_decay)
            self.fc         = L1(self.fc,         self.l1_decay)
        
        self.ranker = None
        
        if self.global_rank == 0:
            logger.info("[SPECTRE] Initialized")
        
    def forward(self, batch, batch_idx=None, return_representations=False):
        B = next(iter(batch.values())).size(0)
        all_points = []
        all_masks = []

        for m, x in batch.items():
            if m in ("mw", "elem_idx", "elem_cnt"):
                continue

            # x: (B, L, D_in)
            B, L, D_in = x.shape
            if L == 0:
                continue
            mask = (x.abs().sum(-1) == 0)  # (B, L), True for padding

            # 1. Encode and reshape
            x_flat   = x.view(B * L, D_in)
            enc_flat = self.encoders[m](x_flat)
            enc_seq  = enc_flat.view(B, L, self.dim_model)  # (B, L, D)

            # 2. Self-attention per modality
            attended = self.self_attn[m](enc_seq, src_key_padding_mask=mask)

            # 3. Accumulate
            all_points.append(attended)
            all_masks.append(mask)

        # 4. Add molecular weight as a 1-point modality
        if "mw" in batch:
            mw_feat = self.mw_embed(batch["mw"].unsqueeze(-1)).unsqueeze(1)  # (B, 1, D)
            all_points.append(mw_feat)
            all_masks.append(torch.zeros(B, 1, dtype=torch.bool, device=mw_feat.device))

        if 'elem_idx' in batch and 'elem_cnt' in batch:
            # (B, L_elem)
            eidx = batch['elem_idx']
            ecnt = batch['elem_cnt'].unsqueeze(-1).float()  # → (B, L, 1)

            # embed
            ve = self.elem_embed(eidx)    # → (B, L, D)
            vc = self.cnt_embed(ecnt)     # → (B, L, D)
            te = ve + vc                  # combine by addition

            # padding mask: True=pad, False=real
            me = (eidx == 0)              # → (B, L)
            all_points.append(te)
            all_masks.append(me)

        joint_seq = torch.cat(all_points, dim=1)  # (B, N_total+1, D)
        joint_mask = torch.cat(all_masks, dim=1)  # (B, N_total+1)

        # 6. Cross-attend from global CLS token
        global_token = self.global_cls.expand(B, 1, -1)  # (B, 1, D)
        for block in self.cross_blocks:
            global_token = block(
                query=global_token,
                key=joint_seq,
                value=joint_seq,
                key_padding_mask=joint_mask
            )

        # 7. Final projection
        out = self.fc(global_token.squeeze(1))  # (B, out_dim)

        if return_representations:
            return global_token.squeeze(1).detach().cpu().numpy()
        return out

    def training_step(self, batch, batch_idx):
        batch_inputs, fps = batch
        logits = self.forward(batch_inputs)
        loss = self.loss(logits, fps)
        self.log("tr/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_inputs, fps = batch
        logits = self.forward(batch_inputs)
        loss = self.loss(logits, fps)
        metrics, _ = cm(
            logits, fps, self.ranker, loss, self.loss, thresh=0.0, 
            query_idx_in_rankingset=batch_idx, no_ranking = True
        )
        self.validation_step_outputs.append(metrics)
        return metrics
    
    def test_step(self, batch, batch_idx):
        batch_inputs, fps = batch
        logits = self.forward(batch_inputs)
        loss = self.loss(logits, fps)
        metrics, _ = cm(
            logits, fps, self.ranker, loss, self.loss, thresh=0.0, 
            query_idx_in_rankingset=batch_idx, no_ranking = False
        )
        self.test_step_outputs.append(metrics)
        return metrics


    def predict_step(self, batch, batch_idx, return_representations=False):
        raise NotImplementedError()

    def on_validation_epoch_end(self):
        feats = self.validation_step_outputs[0].keys()
        di = {}
        for feat in feats:
            di[f"val/mean_{feat}"] = np.mean([v[feat]
                                             for v in self.validation_step_outputs])
        for k, v in di.items():
            self.log(k, v, on_epoch=True, prog_bar=k=="val/mean_rank_1", sync_dist=True)
        self.validation_step_outputs.clear()
        
    def on_test_epoch_end(self):
        feats = self.test_step_outputs[0].keys()
        di = {}
        for feat in feats:
            di[f"test/mean_{feat}"] = np.mean([v[feat]
                                             for v in self.test_step_outputs])
        for k, v in di.items():
            self.log(k, v, on_epoch=True)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        if not self.scheduler:
            return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay = self.weight_decay)
        elif self.scheduler == "attention":
            optim = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay = self.weight_decay, 
                betas=(0.9, 0.98),
                eps=1e-9
            )
            scheduler = NoamOpt(self.dim_model, self.warm_up_steps, optim, self.noam_factor)
            return {
                "optimizer": optim,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
            }

    def log(self, name, value, *args, **kwargs):
        if kwargs.get('sync_dist') is None:
            kwargs['sync_dist'] = logger_should_sync_dist
        super().log(name, value, *args, **kwargs)

    def setup_ranker(self):
        self.ranker = RankingSet(store=self.fp_loader.load_rankingset())