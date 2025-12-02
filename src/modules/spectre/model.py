import math
import logging
from typing import Optional, Tuple
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.distributed as dist
from torchmetrics import MeanMetric
import numpy as np

from .args import SPECTREArgs

from ..data.fp_loader import EntropyFPLoader
from ..data.encoder import build_encoder
from ..core.metrics import cm
from ..core.ranker import RankingSet
from ..core.const import NON_SPECTRAL_INPUTS
from ..loss import BCECosineHybridLoss

logger = logging.getLogger("lightning")

if dist.is_initialized():
    rank = dist.get_rank()
    if rank != 0:
        logger.setLevel(logging.WARNING)


class SPECTRE(pl.LightningModule):
    def __init__(self, args: SPECTREArgs, fp_loader: EntropyFPLoader):
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

        self.scheduler = args.scheduler
        self.dim_model = args.dim_model

        self.use_jaccard = args.use_jaccard
        self.spectral_types = [
            m for m in self.args.input_types if m not in NON_SPECTRAL_INPUTS]

        self.ranker = None
        self.freeze_weights = args.freeze_weights

        self.enc_nmr = build_encoder(
            args.dim_model,
            args.nmr_dim_coords,
            [args.c_wavelength_bounds, args.h_wavelength_bounds],
            args.use_peak_values,
            args.nmr_is_sign_encoding
        )
        self.enc_ms = build_encoder(
            args.dim_model,
            args.ms_dim_coords,
            [args.mz_wavelength_bounds, args.intensity_wavelength_bounds],
            args.use_peak_values,
            args.ms_is_sign_encoding
        )
        self.enc_mw = build_encoder(
            args.dim_model,
            args.mw_dim_coords,
            [args.mw_wavelength_bounds],
            args.use_peak_values,
            args.mw_is_sign_encoding
        )
        self.encoder_list = [self.enc_nmr, self.enc_nmr,
                             self.enc_nmr, self.enc_mw, self.enc_ms]
        if self.global_rank == 0:
            logger.info(f"[SPECTRE] Using {str(self.enc_nmr.__class__)}")

        self.loss = BCECosineHybridLoss(lambda_bce=args.lambda_hybrid)

        # additional nn modules
        self._val_mm = torch.nn.ModuleDict()
        self._test_mm = torch.nn.ModuleDict()

        self.embedding = nn.Embedding(6, self.dim_model)
        self.fc = nn.Linear(self.dim_model, self.out_dim)
        self.latent = torch.nn.Parameter(torch.randn(1, 1, self.dim_model))

        layer = torch.nn.TransformerEncoderLayer(
            d_model=self.dim_model,
            nhead=self.heads,
            dim_feedforward=self.ff_dim,
            batch_first=True,
            dropout=self.dropout,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=self.layers,
        )
        if self.freeze_weights:
            for parameter in self.parameters():
                parameter.requires_grad = False

        if self.global_rank == 0:
            logger.info("[SPECTRE] Initialized")

    def _get_metric_mm(self, store: nn.ModuleDict, feat: str, input_type: str, sync_on_compute: bool = True) -> MeanMetric:
        key = f"{feat}__{input_type}"
        if key not in store:
            store[key] = MeanMetric(
                sync_on_compute=sync_on_compute).to(self.device)
        return store[key]

    def encode(
        self,
        x: torch.Tensor,
        type_indicator: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: Tensor of shape (B, N, D)
        type_indicator: Tensor of shape (B, N) — per peak
        Returns:
            - out: (B, N+1, dim_model)
            - mask: (B, N+1)
        """
        B, N, D = x.shape
        device = x.device
        dim_model = self.latent.shape[-1]

        if mask is None:
            zeros = ~x.sum(dim=2).bool()  # (B, N)
            prefix_mask = torch.zeros((B, 1), dtype=torch.bool, device=device)
            mask = torch.cat([prefix_mask, zeros], dim=1)  # (B, N+1)
        x_flat = x.reshape(B * N, D)
        type_flat = type_indicator.reshape(B * N)
        points_flat = torch.zeros((B * N, dim_model), device=device)

        for type_val, encoder in enumerate(self.encoder_list):
            idx = type_flat == type_val  # (B*N,)
            if idx.any():
                points_flat[idx] = encoder(x_flat[idx])
        points = points_flat.reshape(B, N, dim_model)
        type_embed = self.embedding(type_indicator)  # (B, N, dim_model)
        points += type_embed
        latent = self.latent.expand(B, 1, -1)
        points = torch.cat([latent, points], dim=1)
        out = self.transformer_encoder(points, src_key_padding_mask=mask)
        return out, mask

    def forward(
        self,
        inputs: torch.Tensor,
        type_indicator: torch.Tensor,
        return_representations: bool = False
    ) -> torch.Tensor:
        # (b_s, seq_len, dim_model)
        out, _ = self.encode(inputs, type_indicator)
        # extracts cls token : (b_s, dim_model) -> (b_s, out_dim)
        out_cls = self.fc(out[:, :1, :].squeeze(1))
        if return_representations:
            return out.detach().cpu().numpy()
        return out_cls

    def training_step(self, batch, batch_idx):
        inputs, labels, type_indicator = batch
        out = self.forward(inputs, type_indicator)
        loss = self.loss(out, labels)

        self.log("tr/loss", loss, prog_bar=True,
                 on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, labels, type_indicator = batch
        out = self.forward(inputs, type_indicator)
        loss = self.loss(out, labels)
        metrics, _ = cm(out, labels, self.ranker, loss,
                        self.loss, no_ranking=True)
        input_type_key = "all_inputs" if (dataloader_idx is None or dataloader_idx == 0) \
            else self.spectral_types[dataloader_idx - 1]
        for feat, val in metrics.items():
            mm = self._get_metric_mm(
                self._val_mm, feat, input_type_key, sync_on_compute=True)
            mm.update(torch.tensor(
                val, device=self.device, dtype=torch.float32))

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, labels, type_indicator = batch
        out = self.forward(inputs, type_indicator)
        loss = self.loss(out, labels)

        metrics, _ = cm(out, labels, self.ranker, loss,
                        self.loss, no_ranking=False)
        input_type_key = "all_inputs" if (dataloader_idx is None or dataloader_idx == 0) \
            else self.spectral_types[dataloader_idx - 1]
        for feat, val in metrics.items():
            mm = self._get_metric_mm(
                self._test_mm, feat, input_type_key, sync_on_compute=False)
            mm.update(torch.tensor(
                val, device=self.device, dtype=torch.float32))

    def predict_step(self, batch, batch_idx, return_representations=False):
        raise NotImplementedError()

    def on_validation_epoch_end(self):
        keys = list(self._val_mm.keys())
        if not keys:
            return

        input_types = sorted({k.split("__", 1)[1] for k in keys})
        feats = sorted({k.split("__", 1)[0] for k in keys})

        di = {}
        for feat in feats:
            vals_for_avg = []
            for input_type in input_types:
                mm = self._get_metric_mm(
                    self._val_mm, feat, input_type, sync_on_compute=True)
                v = mm.compute().item()
                di[f"val/mean_{feat}/{input_type}"] = v
                vals_for_avg.append(v)
            di[f"val/mean_{feat}"] = float(np.average(vals_for_avg))

        for k, v in di.items():
            self.log(k, v, on_epoch=True, on_step=False, sync_dist=True)

        for mm in self._val_mm.values():
            mm.reset()

    def on_test_epoch_end(self):
        print("1")
        keys = list(self._test_mm.keys())
        if not keys:
            return
        input_types = sorted({k.split("__", 1)[1] for k in keys})
        feats = sorted({k.split("__", 1)[0] for k in keys})
        print("2")
        di = {}
        for feat in feats:
            vals_for_avg = []
            for input_type in input_types:
                print("3")
                mm = self._get_metric_mm(
                    self._test_mm, feat, input_type, sync_on_compute=False)
                print("4")
                v = mm.compute().item()
                di[f"test/mean_{feat}/{input_type}"] = v
                vals_for_avg.append(v)
            di[f"test/mean_{feat}"] = float(np.average(vals_for_avg))
        print("5")
        for k, v in di.items():
            self.log(k, v, on_epoch=True, on_step=False)
        print("6")
        for mm in self._test_mm.values():
            mm.reset()
        print("7")

    def configure_optimizers(self):
        if not self.scheduler:
            return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.scheduler == "cosine":
            opt = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                    weight_decay=self.weight_decay, betas=(0.9, 0.95))
            total_steps = self.trainer.estimated_stepping_batches
            steps_per_epoch = max(1, total_steps // self.trainer.max_epochs)
            warmup_steps = int((self.args.epochs // 10) *
                               steps_per_epoch) if self.args.warmup else 0

            min_factor = self.args.eta_min / self.args.lr

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
        else:
            raise NotImplementedError(
                f"Scheduler {self.scheduler} not implemented for SPECTRE")

    def setup_ranker(self):
        store = self.fp_loader.load_rankingset(self.args.fp_type)
        self.ranker = RankingSet(store=store, metric="cosine")
