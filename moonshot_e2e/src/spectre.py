import logging
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.distributed as dist
from collections import defaultdict
import numpy as np

from .const import ELEM2IDX
from .settings import Args
from .utils import L1
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
    """
    Single cross‑attention + feed‑forward block.
    Query attends to Key/Value (the spectral peaks).
    """
    def __init__(self, dim_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim_model, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True    # so inputs are (B, L, D)
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
        # query: (B, Q, D); key/value: (B, S, D)
        attn_out, _ = self.attn(
            query=query, 
            key=key, 
            value=value, 
            key_padding_mask=key_padding_mask
        )
        # residual + norm
        q1 = self.norm1(query + attn_out)
        # feed‑forward + norm
        ff_out = self.ff(q1)
        out  = self.norm2(q1 + ff_out)
        return out

class SPECTRE(pl.LightningModule):
    def __init__(self, args: Args, fp_loader = None):
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


        self.bce_pos_weight = None
        logger.info("[SPECTRE] bce_pos_weight = None")

        self.loss = nn.BCEWithLogitsLoss(pos_weight=self.bce_pos_weight)
        self.compute_metric_func = cm

        # additional nn modules 
        self.validation_step_outputs = []
        self.training_step_outputs = []
        self.test_step_outputs = []
        
        if self.freeze_weights:
            for parameter in self.parameters():
                parameter.requires_grad = False
        
        if self.l1_decay > 0:
            self.cross_attn = L1(self.cross_attn, self.l1_decay)
            self.self_attn  = L1(self.self_attn,  self.l1_decay)
            self.fc         = L1(self.fc,         self.l1_decay)

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
        metrics, _ = self.compute_metric_func(
            logits, fps, self.ranker, loss, self.loss, thresh=0.0, 
            rank_by_soft_output=self.rank_by_soft_output,
            query_idx_in_rankingset=batch_idx,
            use_jaccard = self.use_jaccard,
            no_ranking = True
            )
        if type(self.validation_step_outputs) == list: # adapt for child class: optional_input_ranked_transformer
            self.validation_step_outputs.append(metrics)
        return metrics
    
    def test_step(self, batch, batch_idx):
        # 1) unpack
        batch_inputs, fps = batch

        # 2) forward + loss
        logits = self.forward(batch_inputs)
        loss = self.loss(logits, fps)

        # 3) extract per-sample MW values from batch_inputs["mw"]
        mw_list = None
        if "mw" in batch_inputs:
            # now batch_inputs["mw"] is a (B,) tensor of floats
            mw_list = batch_inputs["mw"].tolist()

        # 4) metrics + ranking
        metrics, rank_res = self.compute_metric_func(
            logits, fps, self.ranker, loss, self.loss,
            thresh=0.0,
            rank_by_soft_output=self.rank_by_soft_output,
            query_idx_in_rankingset=batch_idx,
            use_jaccard=self.use_jaccard,
        )
        ranks = rank_res.cpu().tolist()

        # 5) record for on_test_epoch_end
        if isinstance(self.test_step_outputs, list):
            self.test_step_outputs.append(metrics)

        # 6) return exactly the same tuple you expect downstream
        return metrics, mw_list, ranks


    def predict_step(self, batch, batch_idx, return_representations=False):
        raise NotImplementedError()
        
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
    def __init__(self, args: Args, combinations_names: list[str], fp_loader = None):
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