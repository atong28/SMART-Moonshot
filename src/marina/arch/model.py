import logging
import pytorch_lightning as pl
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from collections import defaultdict
import numpy as np

from ..core.const import NON_SPECTRAL_INPUTS
from ..core.settings import MARINAArgs
from ..core.metrics import cm
from ..core.ranker import RankingSet
from ..data.fp_loader import FPLoader
from ..data.encoder import build_encoder
from .attention import MultiHeadAttentionCore
from .loss import BCECosineHybridLoss

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
    def __init__(self, args: MARINAArgs, fp_loader: FPLoader):
        super().__init__()
        
        self.args = args
        self.fp_loader = fp_loader
        if self.global_rank == 0:
            logger.info("[SPECTRE] Started Initializing")
        self.fp_length = args.out_dim
        self.out_dim = args.out_dim
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.heads = args.heads
        self.layers = args.layers
        self.ff_dim = args.ff_dim
        self.dropout = args.dropout
        self.spectral_types = [m for m in self.args.input_types if m not in NON_SPECTRAL_INPUTS]
        self.scheduler = args.scheduler
        self.dim_model = args.dim_model
        self.use_jaccard = args.use_jaccard
        self.freeze_weights = args.freeze_weights

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
        self.loss = BCECosineHybridLoss(lambda_bce = args.lambda_hybrid)
        self.validation_step_outputs = defaultdict(list)
        self.test_step_outputs = defaultdict(list)
        if self.freeze_weights:
            for parameter in self.parameters():
                parameter.requires_grad = False
        self.ranker = None
        spectra_types = set(self.args.input_types) - NON_SPECTRAL_INPUTS
        if self.args.hybrid_early_stopping:
            self.loss_weights = np.array([0.5] + [0.5/len(spectra_types)] * len(spectra_types))
        else:
            self.loss_weights = np.array([1.0] + [0.0] * len(spectra_types))
        if self.global_rank == 0:
            logger.info("[SPECTRE] Initialized")
        
    def forward(self, batch, batch_idx=None, return_representations=False):
        B = next(iter(batch.values())).size(0)
        all_points = []
        all_masks = []
        for m, x in batch.items():
            if m in NON_SPECTRAL_INPUTS:
                continue
            B, L, D_in = x.shape
            mask = (x.abs().sum(-1) == 0)
            x_flat   = x.view(B * L, D_in)
            enc_flat = self.encoders[m](x_flat)
            enc_seq  = enc_flat.view(B, L, self.dim_model)
            attended = self.self_attn[m](enc_seq, src_key_padding_mask=mask)
            all_points.append(attended)
            all_masks.append(mask)
        if "mw" in batch:
            mw_feat = self.mw_embed(batch["mw"].unsqueeze(-1)).unsqueeze(1)
            all_points.append(mw_feat)
            all_masks.append(torch.zeros(B, 1, dtype=torch.bool, device=mw_feat.device))
        joint_seq = torch.cat(all_points, dim=1)
        joint_mask = torch.cat(all_masks, dim=1)
        global_token = self.global_cls.expand(B, 1, -1)
        for block in self.cross_blocks:
            global_token = block(
                query=global_token,
                key=joint_seq,
                value=joint_seq,
                key_padding_mask=joint_mask
            )
        out = self.fc(global_token.squeeze(1))
        if return_representations:
            return global_token.squeeze(1).detach().cpu().numpy()
        return out

    def training_step(self, batch, batch_idx):
        batch_inputs, fps = batch
        logits = self.forward(batch_inputs)
        loss = self.loss(logits, fps)
        self.log("tr/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        batch_inputs, fps = batch
        logits = self.forward(batch_inputs)
        loss = self.loss(logits, fps)
        metrics, _ = cm(
            logits, fps, self.ranker, loss, self.loss,
            thresh=0.0, no_ranking=True
        )
        input_type_key = "all_inputs"
        if dataloader_idx is not None and dataloader_idx > 0:
            input_type_key = self.spectral_types[dataloader_idx - 1]
        self.validation_step_outputs[input_type_key].append(metrics)
        return metrics
    
    def test_step(self, batch, batch_idx, dataloader_idx=None):
        batch_inputs, fps = batch
        logits = self.forward(batch_inputs)
        loss = self.loss(logits, fps)
        metrics, _ = cm(
            logits, fps, self.ranker, loss, self.loss,
            thresh=0.0, no_ranking=False
        )
        input_type_key = "all_inputs"
        if dataloader_idx is not None and dataloader_idx > 0:
            input_type_key = self.spectral_types[dataloader_idx - 1]
        self.test_step_outputs[input_type_key].append(metrics)
        return metrics

    def predict_step(self, batch, batch_idx, return_representations=False):
        raise NotImplementedError()

    def on_validation_epoch_end(self):
        feats = self.validation_step_outputs["all_inputs"][0].keys()
        di = {}
        for feat in feats:
            for input_type in self.validation_step_outputs.keys():
                di[f"val/mean_{feat}/{input_type}"] = np.mean(
                    [v[feat] for v in self.validation_step_outputs[input_type]]
                )
            di[f"val/mean_{feat}"] = np.average([di[f"val/mean_{feat}/{input_type}"] for input_type in self.validation_step_outputs.keys()], weights=self.loss_weights)
        for k, v in di.items():
            self.log(k, v, on_epoch=True, sync_dist=True)
        self.validation_step_outputs.clear()
        
    def on_test_epoch_end(self):
        feats = self.test_step_outputs["all_inputs"][0].keys()
        di = {}
        for feat in feats:
            for input_type in self.test_step_outputs.keys():
                di[f"test/mean_{feat}/{input_type}"] = np.mean(
                    [v[feat] for v in self.test_step_outputs[input_type]]
                )
            di[f"test/mean_{feat}"] = np.average([di[f"test/mean_{feat}/{input_type}"] for input_type in self.test_step_outputs.keys()], weights=self.loss_weights)
        for k, v in di.items():
            self.log(k, v, on_epoch=True, sync_dist=True)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        if not self.scheduler:
            return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay = self.weight_decay)
        elif self.scheduler == "cosine":
            opt = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay, betas=(0.9, 0.95))
            total_steps = self.trainer.estimated_stepping_batches
            steps_per_epoch = max(1, total_steps // self.trainer.max_epochs)
            warmup_steps = int((self.args.epochs // 10) * steps_per_epoch)

            min_factor = self.args.eta_min / self.args.lr  # final LR as a fraction of base LR

            def lr_lambda(step: int):
                if step < warmup_steps:
                    return max(1e-6, step / max(1, warmup_steps))
                t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                cosine = 0.5 * (1.0 + math.cos(math.pi * t))
                return min_factor + (1 - min_factor) * cosine

            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": sched,
                    "interval": "step",  # step every optimizer step
                    "name": "lr",
                },
            }

    def log(self, name, value, *args, **kwargs):
        if kwargs.get('sync_dist') is None:
            kwargs['sync_dist'] = logger_should_sync_dist
        super().log(name, value, *args, **kwargs)

    def setup_ranker(self):
        store = self.fp_loader.load_rankingset(self.args.fp_type)
        self.ranker = RankingSet(store=store, metric="cosine")

