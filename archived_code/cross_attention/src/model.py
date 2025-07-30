import logging
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.distributed as dist
from collections import defaultdict
import numpy as np

from .settings import Args
from .utils import L1
from .fp_loaders.entropy import EntropyFPLoader
from .encoder import build_encoder
from .metrics import cm
from .ranker import RankingSet
from .lr_scheduler import NoamOpt

logger = logging.getLogger("lightning")
if dist.is_initialized():
    rank = dist.get_rank()
    if rank != 0:
        logger.setLevel(logging.WARNING)
logger_should_sync_dist = torch.cuda.device_count() > 1

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim_model, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim_model)
        self.ff = nn.Sequential(
            nn.Linear(dim_model, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim_model),
        )
        self.norm2 = nn.LayerNorm(dim_model)

    def forward(self, query, key, value, key_padding_mask=None, return_attn_weights=False):
        attn_out, attn_weights = self.attn(
            query=query, 
            key=key, 
            value=value, 
            key_padding_mask=key_padding_mask,
            need_weights=return_attn_weights,
            average_attn_weights=False  # if you want separate weights per head
        )
        q1 = self.norm1(query + attn_out)
        ff_out = self.ff(q1)
        out  = self.norm2(q1 + ff_out)

        if return_attn_weights:
            return out, attn_weights
        return out

class SPECTRE(pl.LightningModule):
    def __init__(self, args: Args, fp_loader: EntropyFPLoader):
        super().__init__()
        
        self.args = args
        self.fp_loader = fp_loader
        
        if self.global_rank == 0:
            logger.info("[SPECTRE] Started Initializing")

        self.fp_length = args.out_dim
        self.out_dim = args.out_dim
        
        self.batch_size = args.batch_size
        self.num_class = None # TODO: delete
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
        
        # don't set ranking set if you just want to treat it as a module
        self.fp_type=args.fp_type
        self.rank_by_soft_output = args.rank_by_soft_output
        self.rank_by_test_set = args.rank_by_test_set
        
        self.fp_radius = args.fp_radius
        self.ranker = None
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
            args.ms_id_dim_coords,
            [args.mz_wavelength_bounds, args.intensity_wavelength_bounds],
            args.use_peak_values
        )
        self.enc_id = build_encoder(
            args.dim_model,
            args.ms_id_dim_coords,
            [args.id_wavelength_bounds, args.abundance_wavelength_bounds],
            args.use_peak_values
        )
        self.enc_mw = build_encoder(
            args.dim_model,
            args.mw_dim_coords,
            [args.mw_wavelength_bounds],
            args.use_peak_values
        )
        self.encoder_list = [self.enc_nmr, self.enc_nmr, self.enc_nmr, self.enc_mw, self.enc_id, self.enc_ms]
        if self.global_rank == 0:
            logger.info(f"[SPECTRE] Using {str(self.enc_nmr.__class__)}")

        self.bce_pos_weight = None
        logger.info("[SPECTRE] bce_pos_weight = None")

        self.loss = nn.BCEWithLogitsLoss(pos_weight=self.bce_pos_weight)
        self.compute_metric_func = cm

        # additional nn modules 
        self.validation_step_outputs = []
        self.training_step_outputs = []
        self.test_step_outputs = []

        self.embedding = nn.Embedding(6, self.dim_model)
        self.fc = nn.Linear(self.dim_model, self.out_dim)
        self.latent = torch.nn.Parameter(torch.randn(1, 1, self.dim_model))

        self.cross_blocks = nn.ModuleList([
            CrossAttentionBlock(
                dim_model=self.dim_model,
                num_heads=self.heads,
                ff_dim=self.ff_dim,
                dropout=self.dropout
            )
            for _ in range(self.layers)
        ])
        
        if self.freeze_weights:
            for parameter in self.parameters():
                parameter.requires_grad = False
        if self.l1_decay > 0:
            self.cross_blocks = L1(self.cross_blocks, self.l1_decay)
            self.fc = L1(self.fc, self.l1_decay)

        if self.global_rank == 0:
            logger.info("[SPECTRE] Initialized")

    def encode(self, x, type_indicator, mask=None):
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
        type_embed = self.embedding(type_indicator)
        points = points + type_embed               # (B, N, D)
        latent = self.latent.expand(B, 1, -1)      # (B, 1, D)

        # Now run L cross‑attention blocks:
        for block in self.cross_blocks:
            # mask[:,1:] masks the padded peaks
            latent = block(
                query=latent, 
                key=points, 
                value=points, 
                key_padding_mask=mask[:,1:]
            )
        return latent, mask

    def forward(self, hsqc, type_indicator, return_representations=False):
        """The forward pass.
        Parameters
        ----------
        hsqc: torch.Tensor of shape (batch_size, n_points, 3)
            The hsqc to embed. Axis 0 represents an hsqc, axis 1
            contains the coordinates in the hsqc, and axis 2 is essentially is
            a 3-tuple specifying the coordinate's x, y, and z value. These
            should be zero-padded, such that all of the hsqc in the batch
            are the same length.
        """
        latent, _ = self.encode(hsqc, type_indicator)
        # latent is (B, 1, D): grab that token
        out_cls = self.fc(latent.squeeze(1))
        if return_representations:
            return latent.squeeze(1).detach().cpu().numpy()
        return out_cls
    
    def forward_with_attention(self, x, type_indicator):
        """
        Returns the attention weights from each CrossAttentionBlock.
        """
        B, N, D = x.shape
        device = x.device
        dim_model = self.latent.shape[-1]

        zeros = ~x.sum(dim=2).bool()  # (B, N)
        prefix_mask = torch.zeros((B, 1), dtype=torch.bool, device=device)
        mask = torch.cat([prefix_mask, zeros], dim=1)  # (B, N+1)

        x_flat = x.reshape(B * N, D)
        type_flat = type_indicator.reshape(B * N)
        points_flat = torch.zeros((B * N, dim_model), device=device)

        for type_val, encoder in enumerate(self.encoder_list):
            idx = type_flat == type_val
            if idx.any():
                points_flat[idx] = encoder(x_flat[idx])
        points = points_flat.reshape(B, N, dim_model)
        type_embed = self.embedding(type_indicator)
        points = points + type_embed
        latent = self.latent.expand(B, 1, -1)

        attention_weights = []

        for block in self.cross_blocks:
            latent, attn_weights = block(
                query=latent,
                key=points,
                value=points,
                key_padding_mask=mask[:, 1:],
                return_attn_weights=True
            )
            attention_weights.append(attn_weights.detach().cpu())  # shape: (B, num_heads, 1, N)

        return attention_weights, mask[:, 1:].cpu(), type_indicator.cpu()


    def training_step(self, batch, batch_idx):
        
        inputs, labels, NMR_type_indicator = batch
        out = self.forward(inputs, NMR_type_indicator)
        loss = self.loss(out, labels)
        
        self.log("tr/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels, NMR_type_indicator = batch
        out = self.forward(inputs, NMR_type_indicator)
        loss = self.loss(out, labels)
        metrics, _ = self.compute_metric_func(
            out, labels, self.ranker, loss, self.loss, thresh=0.0, 
            rank_by_soft_output=self.rank_by_soft_output,
            query_idx_in_rankingset=batch_idx,
            use_jaccard = self.use_jaccard,
            no_ranking = True
            )
        
        if type(self.validation_step_outputs) == list: # adapt for child class: optional_input_ranked_transformer
            self.validation_step_outputs.append(metrics)
        return metrics
    
    def test_step(self, batch, batch_idx):
        inputs, labels, NMR_type_indicator = batch
        out = self.forward(inputs, NMR_type_indicator)
        loss = self.loss(out, labels)
        
        mw_list = None
        if 'mw' in self.args.input_types:
            mw_list = []
            for x, t in zip(inputs, NMR_type_indicator):
                mw_values = x[t == 3]  # MW_TYPE = 3
                mw_list.append(mw_values[0, 0].item() if len(mw_values) > 0 else None)
        
        metrics, rank_res = self.compute_metric_func(
            out, labels, self.ranker, loss, self.loss, thresh=0.0,
            rank_by_soft_output=self.rank_by_soft_output,
            query_idx_in_rankingset=batch_idx,
            use_jaccard = self.use_jaccard,
        )
        ranks = rank_res.cpu().tolist()
        if type(self.test_step_outputs)==list:
            self.test_step_outputs.append(metrics)
        return metrics, mw_list, ranks

    def predict_step(self, batch, batch_idx, return_representations=False):
        x, smiles_chemical_name = batch
        if return_representations:
            return self.forward(x, return_representations=True)
        out = self.forward(x)
        preds = torch.sigmoid(out)
        top_k_idxs = self.ranker.retrieve_idx(preds)
        return top_k_idxs
        
    def on_train_epoch_end(self):
        if self.training_step_outputs:
            feats = self.training_step_outputs[0].keys()
            di = {}
            for feat in feats:
                di[f"tr/mean_{feat}"] = np.mean([v[feat]
                                                for v in self.training_step_outputs])
            for k, v in di.items():
                self.log(k, v, on_epoch=True)
            self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        feats = self.validation_step_outputs[0].keys()
        di = {}
        for feat in feats:
            di[f"val/mean_{feat}"] = np.mean([v[feat]
                                             for v in self.validation_step_outputs])
        for k, v in di.items():
            self.log(k, v, on_epoch=True, prog_bar=k=="val/mean_rank_1")
        self.validation_step_outputs.clear()
        
    def on_test_epoch_end(self):
        feats = self.test_step_outputs[0].keys()
        di = {}
        for feat in feats:
            di[f"test/mean_{feat}"] = np.mean([v[feat]
                                             for v in self.test_step_outputs])
        for k, v in di.items():
            self.log(k, v, on_epoch=True)
            # self.log(k, v, on_epoch=True)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        if not self.scheduler:
            return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay = self.weight_decay)
        elif self.scheduler == "attention":
            optim = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay = self.weight_decay, 
                                     betas=(0.9, 0.98), eps=1e-9)
            
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
        use_hyun_fp = self.fp_type == "HYUN"
        self.ranker = RankingSet(
            store=self.fp_loader.build_inference_ranking_set_with_everything(
                fp_dim = self.fp_length, 
                max_radius = self.fp_radius,
                use_hyun_fp=use_hyun_fp
            ),
            batch_size=self.batch_size, CE_num_class=self.num_class, need_to_normalize=False
        )

class OptionalInputSPECTRE(SPECTRE):
    def __init__(self, args: Args, fp_loader: EntropyFPLoader, combinations_names: list[str]):
        super().__init__(args, fp_loader)
        self.validation_step_outputs = defaultdict(list)
        self.test_step_outputs = defaultdict(list)
        self.all_dataset_names = combinations_names
        self.loader_idx = None
        self.validate_all = args.validate_all
            
    def validation_step(self, batch, batch_idx, dataloader_idx = None):
        if not self.validate_all:
            current_batch_name = 'ALL'
        else:
            current_batch_name = self.all_dataset_names[dataloader_idx]
        metrics = super().validation_step(batch, batch_idx)
        self.validation_step_outputs[current_batch_name].append(metrics)
        return metrics
    
    def test_step(self, batch, batch_idx, dataloader_idx = 0):
        if not self.validate_all:
            current_batch_name = 'ALL'
        else:
            current_batch_name = self.all_dataset_names[dataloader_idx]
        metrics, mw_list, ranks = super().test_step(batch, batch_idx)
        if not hasattr(self, "mw_rank_records"):
            self.mw_rank_records = {name: [] for name in self.all_dataset_names}
        for mw, rank in zip(mw_list, ranks):
            self.mw_rank_records[current_batch_name].append({
                "mw": mw,
                "rank_1": int(rank < 1),
                "rank_5": int(rank < 5),
                "rank_10": int(rank < 10),
            })
        self.test_step_outputs[current_batch_name].append(metrics)

        return metrics
    
    def predict_step(self, batch, batch_idx, dataloader_idx, return_representations=False):
        return super().predict_step(batch, batch_idx, return_representations)
    
    def on_validation_epoch_end(self):
        total_features = defaultdict(list)
        for dataset_name in self.all_dataset_names:
            if len(self.validation_step_outputs[dataset_name]) == 0:
                continue
            feats = self.validation_step_outputs[dataset_name][0].keys()
            di = {}
            for feat in feats:
                curr_dataset_curr_feature = np.mean([v[feat] for v in self.validation_step_outputs[dataset_name]])
                if dataset_name == 'ALL':
                    di[f"val/mean_{feat}"] = curr_dataset_curr_feature
                else:
                    di[f"val/mean_{feat}/{dataset_name}"] = curr_dataset_curr_feature
                total_features[feat].append(curr_dataset_curr_feature)
            for k, v in di.items():
                self.log(k, v, on_epoch=True, prog_bar="rank_1/" in k)
        self.validation_step_outputs.clear()
        
        
    def on_test_epoch_end(self):
        total_features = defaultdict(list)
        for dataset_name in self.all_dataset_names:
            if len(self.test_step_outputs[dataset_name]) == 0:
                continue
            feats = self.test_step_outputs[dataset_name][0].keys()
            di = {}
            for feat in feats:
                curr_dataset_curr_feature = np.mean([v[feat] for v in self.test_step_outputs[dataset_name]])
                di[f"test/mean_{feat}/{dataset_name}"] = curr_dataset_curr_feature
                total_features[feat].append(curr_dataset_curr_feature)
            for k, v in di.items():
                self.log(k, v, on_epoch=True, prog_bar="rank_1" in k)
        self.test_step_outputs.clear()

def build_model(args: Args, optional_inputs: bool, fp_loader: EntropyFPLoader, combinations_names = None):
    if optional_inputs:
        logger.info('[SPECTRE] Using optional input ranked transformer')
        return OptionalInputSPECTRE(args, fp_loader, combinations_names)
    logger.info('[SPECTRE] Using fixed input ranked transformer')
    return SPECTRE(args, fp_loader)