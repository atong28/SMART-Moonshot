import logging
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.distributed as dist

from settings import SpectreArgs
from const import ELEM2IDX
from spectre.utils import L1
from spectre.encoder import build_encoder

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
    def __init__(self, args: SpectreArgs):
        super().__init__()
        
        self.args = args
        
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
        
        if self.freeze_weights:
            for parameter in self.parameters():
                parameter.requires_grad = False
        
        if self.l1_decay > 0:
            self.cross_attn = L1(self.cross_attn, self.l1_decay)
            self.self_attn  = L1(self.self_attn,  self.l1_decay)
            self.fc         = L1(self.fc,         self.l1_decay)

        if self.global_rank == 0:
            logger.info("[SPECTRE] Initialized")
        
    def forward(self, batch, batch_idx=None):
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

        return global_token.squeeze(1).detach()