import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim
from torch_geometric.nn import GATConv
from rdkit.Chem import RWMol, Atom, MolToSmiles, SanitizeMol, AllChem, DataStructs
from rdkit.Chem.rdchem import BondType
from rdkit import Chem
import pytorch_lightning as pl
from torch_geometric.data import Data

from .const import ELEM2IDX
from .settings import Args
from .encoder import build_encoder
from .utils import q_sample, get_beta_schedule

logger = logging.getLogger("lightning")


class GraphDenoiser(nn.Module):
    def __init__(self,
                 node_feat_dim: int,   # e.g. 5
                 cond_dim:      int,   # args.dim_model
                 hidden:        int,
                 heads:         int,
                 layers:        int):
        super().__init__()
        self.node_proj = nn.Linear(node_feat_dim, hidden)
        self.time_mlp  = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
        )
        self.cond_mlp  = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
        )
        self.convs     = nn.ModuleList([
            GATConv(hidden, hidden, heads=heads, concat=False, dropout=0.1)
            for _ in range(layers)
        ])
        self.out_lin   = nn.Linear(hidden, node_feat_dim)

    def forward(self, x, edge_index, batch, t, cond):
        # x: [N_nodes, node_feat_dim]
        t_node  = self.time_mlp(t.float().unsqueeze(1))[batch]
        cond_node = self.cond_mlp(cond)[batch]
        h = self.node_proj(x) + t_node + cond_node
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
        return self.out_lin(h)


class SPECTREBackbone(nn.Module):
    """
    Encode spectra + formula + MW → (B, dim_model) cond vector.
    """
    def __init__(self, args: Args):
        super().__init__()
        D = args.dim_model

        # build encoders for each modality
        self.enc_nmr = build_encoder(
            D, args.nmr_dim_coords,
            [args.c_wavelength_bounds, args.h_wavelength_bounds],
            args.use_peak_values
        )
        self.enc_ms = build_encoder(
            D, args.ms_dim_coords,
            [args.mz_wavelength_bounds, args.intensity_wavelength_bounds],
            args.use_peak_values
        )

        encs = {
            "hsqc":     self.enc_nmr,
            "c_nmr":    self.enc_nmr,
            "h_nmr":    self.enc_nmr,
            "mass_spec":self.enc_ms
        }
        self.encoders = nn.ModuleDict({
            m: encs[m] for m in encs if m in args.input_types
        })

        self.self_attn = nn.ModuleDict({
            m: nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=D, nhead=args.heads,
                    dim_feedforward=args.ff_dim,
                    dropout=args.dropout,
                    batch_first=True
                ), num_layers=args.self_attn_layers
            )
            for m in self.encoders
        })

        self.cross_blocks = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=D, num_heads=args.heads,
                dropout=args.dropout, batch_first=True
            )
            for _ in range(args.layers)
        ])
        self.norm1        = nn.LayerNorm(D)
        self.norm2        = nn.LayerNorm(D)

        # extra embeddings
        self.mw_embed      = nn.Linear(1, D)
        self.formula_embed = nn.Linear(len(ELEM2IDX)-1, D)
        num_elem          = len(ELEM2IDX)+1
        self.elem_embed   = nn.Embedding(num_elem, D, padding_idx=0)
        self.cnt_embed    = nn.Linear(1, D)

        self.global_cls   = nn.Parameter(torch.randn(1,1,D))

    def forward(self, batch_inputs):
        B = next(iter(batch_inputs.values())).size(0)
        all_pts, all_masks = [], []

        # 1) each sequence
        for m, x in batch_inputs.items():
            if m in ("mw","formula_vec","elem_idx","elem_cnt"):
                continue
            B_, L, _ = x.shape
            if L == 0:
                continue
            mask  = x.abs().sum(-1)==0
            flat  = x.view(B_*L, -1)
            enc   = self.encoders[m](flat).view(B_,L,-1)
            attn  = self.self_attn[m](enc, src_key_padding_mask=mask)
            all_pts.append(attn); all_masks.append(mask)

        # 2) mw
        if "mw" in batch_inputs:
            mw = batch_inputs["mw"].unsqueeze(-1)
            mw = self.mw_embed(mw).unsqueeze(1)
            all_pts.append(mw)
            all_masks.append(torch.zeros(B,1,device=mw.device,dtype=torch.bool))

        # 4) elem_idx/elem_cnt
        if "elem_idx" in batch_inputs and "elem_cnt" in batch_inputs:
            eidx = batch_inputs["elem_idx"]
            ecnt = batch_inputs["elem_cnt"].unsqueeze(-1).float()
            ve   = self.elem_embed(eidx)
            vc   = self.cnt_embed(ecnt)
            mask = eidx==0
            all_pts.append(ve+vc); all_masks.append(mask)

        # 5) cross‐attention
        seq  = torch.cat(all_pts, dim=1)
        mask = torch.cat(all_masks, dim=1)
        h    = self.global_cls.expand(B,1,-1)
        for attn in self.cross_blocks:
            h2, _ = attn(query=h, key=seq, value=seq, key_padding_mask=mask)
            h   = self.norm1(h+h2)
            h3  = F.relu(h)
            h   = self.norm2(h+h3)

        return h.squeeze(1)  # (B, D)

class Moonshot(pl.LightningModule):
    def __init__(self, args: Args):
        super().__init__()
        self.args = args

        #––– build the SPECTRE backbone for conditioning –––
        self.encoder = SPECTREBackbone(args)

        #––– diffusion schedule –––
        betas  = get_beta_schedule(
            args.timesteps, args.beta_start, args.beta_end
        )
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))

        #––– the denoiser –––
        self.denoiser = GraphDenoiser(
            node_feat_dim = args.node_feat_dim,
            cond_dim      = args.dim_model,
            hidden        = args.diff_hidden,
            heads         = args.diff_heads,
            layers        = args.diff_layers
        )

        #––– atom + bond heads –––
        D = args.node_feat_dim
        self.idx2elem = {idx: e for e,idx in ELEM2IDX.items()}
        self.atom_clf = nn.Linear(D, len(self.idx2elem))
        self.bond_type_map = {
            0: BondType.SINGLE, 1: BondType.DOUBLE,
            2: BondType.TRIPLE, 3: BondType.AROMATIC
        }
        self.bond_clf = nn.Sequential(
            nn.Linear(2*D, args.diff_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(args.diff_hidden, len(self.bond_type_map))
        )
        
        self.formula_cond_mlp = nn.Sequential(
            nn.Linear(len(ELEM2IDX)-1, args.dim_model),
            nn.ReLU(inplace=True),
            nn.Linear(args.dim_model, args.dim_model)
        )

        self.lr = args.lr
    
    @torch.no_grad()
    def _score_molecule(self, pred_smiles: str, ref_smiles: str):
        # 1) parse
        pred_mol = Chem.MolFromSmiles(pred_smiles)
        ref_mol  = Chem.MolFromSmiles(ref_smiles)
        if pred_mol is None or ref_mol is None:
            return {
                "valid": 0.0,
                "exact": 0.0,
                "tanimoto": 0.0
            }

        # 2) exact match?
        can_pred = Chem.MolToSmiles(pred_mol, isomericSmiles=True)
        can_ref  = Chem.MolToSmiles(ref_mol,  isomericSmiles=True)
        exact = 1.0 if can_pred == can_ref else 0.0

        # 3) compute Morgan fingerprints
        fp1 = AllChem.GetMorganFingerprintAsBitVect(pred_mol, radius=2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(ref_mol,  radius=2, nBits=2048)
        tanimoto = DataStructs.TanimotoSimilarity(fp1, fp2)

        return {
            "valid": 1.0,
            "exact": exact,
            "tanimoto": tanimoto
        }

    def forward(self, batch_inputs, graph):
        # 1) encode spectral + MW + elem‐tokens:
        spec_cond = self.encoder(batch_inputs)

        # 2) embed the full formula_vec:
        if "formula_vec" in batch_inputs:
            formula_emb = self.formula_cond_mlp(batch_inputs["formula_vec"])
            cond = spec_cond + formula_emb
        else:
            cond = spec_cond
        x0         = graph.x
        edge_index = graph.edge_index
        batch_idx  = graph.batch
        B          = graph.num_graphs

        t     = torch.randint(0, self.args.timesteps, (B,), device=self.device)
        noise = torch.randn_like(x0)
        x_t   = q_sample(x0, t, noise, self.alphas_cumprod, batch_idx)

        eps = self.denoiser(x_t, edge_index, batch_idx, t, cond)
        return eps, noise

    def training_step(self, batch, batch_idx):
        inp, graph = batch
        eps, noise = self(inp, graph)
        loss = F.mse_loss(eps, noise)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inp, graph = batch
        eps, noise = self(inp, graph)
        loss = F.mse_loss(eps, noise)
        self.log("val/loss", loss, prog_bar=True)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        inputs, graph = batch
        eps, noise = self(inputs, graph)
        loss = F.mse_loss(eps, noise)
        self.log("val/loss", loss, prog_bar=True)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        inputs, graph = batch
        B = graph.num_graphs

        # compute diffusion MSE on the whole batch
        eps, noise = self(inputs, graph)
        loss = F.mse_loss(eps, noise)

        # now molecule‐level metrics, one graph at a time
        all_scores = {"valid": 0.0, "exact": 0.0, "tanimoto": 0.0}
        for i in range(B):
            # slice out the i-th graph
            mask = (graph.batch == i)
            sub_x         = graph.x[mask]
            # sub_edge_index: keep only edges with both ends in this graph
            edge_mask     = graph.batch[graph.edge_index[0]] == i
            sub_edge_index= graph.edge_index[:, edge_mask]
            sub_batch     = torch.zeros(sub_x.size(0), dtype=torch.long, device=self.device)
            sub_graph     = Data(x=sub_x, edge_index=sub_edge_index, batch=sub_batch)
            # slice inputs
            sub_inputs    = {k: v[i:i+1] for k,v in inputs.items()}
            pred_smiles   = self.sample(sub_inputs, sub_graph)  
            ref_smiles    = graph.smiles_list[i]  # you need a list of smiles per graph
            scores        = self._score_molecule(pred_smiles, ref_smiles)
            for k in all_scores:
                all_scores[k] += scores[k]

        # average
        for k,v in all_scores.items():
            all_scores[k] = v / B

        # log
        out = {"test_loss": loss.item(), **all_scores}
        return out

    def on_test_epoch_end(self, outputs):
        # aggregate over all examples
        valid    = torch.tensor([o["valid"]    for o in outputs]).mean()
        exact    = torch.tensor([o["exact"]    for o in outputs]).mean()
        tanimoto = torch.tensor([o["tanimoto"] for o in outputs]).mean()
        self.log("test/validity",    valid)
        self.log("test/exact_match", exact)
        self.log("test/tanimoto",    tanimoto)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)

    @torch.no_grad()
    def sample(self, batch_inputs, graph_template):
        cond       = self.encoder(batch_inputs)
        edge_index = graph_template.edge_index
        batch_idx  = graph_template.batch
        N_nodes    = graph_template.num_nodes

        x = torch.randn(N_nodes, self.args.node_feat_dim, device=self.device)
        for t in reversed(range(self.args.timesteps)):
            # denoise
            eps_p = self.denoiser(x, edge_index, batch_idx,
                                  torch.tensor([t], device=self.device),  # you can pass [t]
                                  cond)

            beta_t      = self.betas[t]                     # scalar tensor
            alpha_t     = 1.0 - beta_t                      # scalar
            alpha_bar_t = self.alphas_cumprod[t]            # scalar

            # scalar coefficients → broadcast
            coef1 = 1.0 / alpha_t.sqrt()                    # scalar
            coef2 =     beta_t  / (1.0 - alpha_bar_t).sqrt()# scalar
            mu     = coef1 * (x - coef2 * eps_p)             # [N_nodes, D]

            if t > 0:
                x = mu + beta_t.sqrt() * torch.randn_like(x)
            else:
                x = mu

        # decode atoms & bonds as before...
        atom_logits = self.atom_clf(x)
        atom_preds  = atom_logits.argmax(-1)

        src, dst   = edge_index
        pair_feats = torch.cat([x[src], x[dst]], dim=-1)
        bond_preds = self.bond_clf(pair_feats).argmax(-1)

        mol = RWMol()
        for idx in atom_preds.tolist():
            mol.AddAtom(Atom(self.idx2elem.get(idx, "C")))
        for (u,v), b in zip(edge_index.t().tolist(), bond_preds.tolist()):
            mol.AddBond(u, v, self.bond_type_map[b])
        SanitizeMol(mol)
        return MolToSmiles(mol)


def build_model(args):
    return Moonshot(args)
