import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim
from torch_geometric.nn import GATConv
from rdkit.Chem import RWMol, Atom, MolToSmiles, SanitizeMol
from rdkit.Chem.rdchem import BondType

from .spectre import OptionalInputSPECTRE
from .settings import Args
from .utils import q_sample, get_beta_schedule
from .const import ELEM2IDX

logger = logging.getLogger(__name__)

class GraphDenoiser(nn.Module):
    """
    Predicts per‐node noise in the diffusion process.
    FiLM‐style time + condition → message‐passing → noise prediction.
    """
    def __init__(self, node_dim: int, cond_dim: int, hidden: int, heads: int, layers: int):
        super().__init__()
        self.node_proj = nn.Linear(node_dim,   hidden)
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
        self.convs = nn.ModuleList([
            GATConv(hidden, hidden, heads=heads, concat=False, dropout=0.1)
            for _ in range(layers)
        ])
        self.out_lin = nn.Linear(hidden, node_dim)

    def forward(self, x, edge_index, batch, t, cond):
        # x: [N_nodes, node_dim]
        # batch: [N_nodes] mapping node→graph
        # t:     [B] timestep per graph
        # cond:  [B, cond_dim] graph‐level
        t_node    = self.time_mlp(t.unsqueeze(1).float())[batch]
        cond_node = self.cond_mlp(cond)[batch]
        h = self.node_proj(x) + t_node + cond_node
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
        return self.out_lin(h)


class MoonshotDiffusion(pl.LightningModule):
    """
    A two‐stage model: 1) OptionalInputSPECTRE backbone → cond vectors
                       2) GraphDenoiser → diffusion on molecular graphs
    """
    def __init__(
        self,
        args: Args,
        combinations_names: list[str],
        fp_loader = None                 # passed through from your usual train/test
    ):
        super().__init__()
        self.args = args

        # 1) pretrained‐style backbone (you can freeze it if you like)
        self.backbone = OptionalInputSPECTRE(args, combinations_names, fp_loader)
        if args.freeze_weights:
            for p in self.backbone.parameters(): p.requires_grad = False

        # 2) diffusion schedule
        betas  = get_beta_schedule(args.timesteps, args.beta_start, args.beta_end)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)                          
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))

        # 3) denoiser network
        self.denoiser = GraphDenoiser(
            node_dim = args.node_feat_dim,
            cond_dim = args.dim_model,
            hidden   = args.diff_hidden,
            heads    = args.diff_heads,
            layers   = args.diff_layers
        )

        # 4) atom & bond classification heads for sampling
        D = args.node_feat_dim
        self.idx2elem = {idx: elem for elem, idx in ELEM2IDX.items()}
        self.atom_clf = nn.Linear(D, len(self.idx2elem))
        self.bond_type_map = {
            0: BondType.SINGLE,
            1: BondType.DOUBLE,
            2: BondType.TRIPLE,
            3: BondType.AROMATIC
        }
        self.bond_clf = nn.Sequential(
            nn.Linear(2 * D, args.diff_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(args.diff_hidden, len(self.bond_type_map))
        )

        self.lr = args.lr

    def forward(self, batch_inputs, graph):
        # 1) get B×D conditioning from the backbone
        cond = self.backbone(batch_inputs, return_representations=True)  # (B, D_cond)

        # 2) pull node‐features + graph structure
        x0         = graph.x            # [N_nodes, node_feat_dim]
        edge_index = graph.edge_index   # [2, N_edges]
        batch_idx  = graph.batch        # [N_nodes]
        B          = graph.num_graphs

        # 3) sample t ∼ Uniform[0, T), noise ∼ N(0, I)
        t     = torch.randint(0, self.args.timesteps, (B,), device=self.device)
        noise = torch.randn_like(x0)

        # 4) diffuse x0 → x_t
        x_t = q_sample(x0, t, noise, self.alphas_cumprod, batch_idx)

        # 5) predict the noise
        eps_pred = self.denoiser(x_t, edge_index, batch_idx, t, cond)
        return eps_pred, noise

    def training_step(self, batch, batch_idx):
        batch_inputs, graph = batch
        eps_pred, noise = self(batch_inputs, graph)
        loss = F.mse_loss(eps_pred, noise)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_inputs, graph = batch
        eps_pred, noise = self(batch_inputs, graph)
        loss = F.mse_loss(eps_pred, noise)
        self.log("val/loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        batch_inputs, graph = batch
        eps_pred, noise = self(batch_inputs, graph)
        loss = F.mse_loss(eps_pred, noise)
        self.log("test/loss", loss, prog_bar=True)
        return {"test_loss": loss}

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)

    @torch.no_grad()
    def sample(self, batch_inputs, graph_template):
        """
        Reverse‐diffuse and decode node‐ and bond‐predictions back to a SMILES.
        Assumes batch_size=1 for simplicity.
        """
        cond       = self.backbone(batch_inputs, return_representations=True)  # [1, D_cond]
        edge_index = graph_template.edge_index
        batch_idx  = graph_template.batch
        N_nodes    = graph_template.num_nodes

        # start from pure noise
        x = torch.randn(N_nodes, self.args.node_feat_dim, device=self.device)

        # reverse diffusion loop
        for t in reversed(range(self.args.timesteps)):
            t_b   = torch.full(( graph_template.num_graphs,), t,
                               dtype=torch.long, device=self.device)
            eps_p = self.denoiser(x, edge_index, batch_idx, t_b, cond)

            β   = self.betas[t]
            α   = 1.0 - β
            ᾱ   = self.alphas_cumprod[t]
            c1  = (1/α.sqrt())[batch_idx].unsqueeze(-1)
            c2  = (β/(1-ᾱ).sqrt())[batch_idx].unsqueeze(-1)
            μ   = c1 * (x - c2 * eps_p)

            if t > 0:
                x = μ + β.sqrt() * torch.randn_like(x)
            else:
                x = μ

        # atom decode
        atom_logits = self.atom_clf(x)            # [N_nodes, num_atom_types]
        atom_preds  = atom_logits.argmax(dim=-1)

        # bond decode
        src, dst     = edge_index
        pair_feats   = torch.cat([x[src], x[dst]], dim=-1)
        bond_preds   = self.bond_clf(pair_feats).argmax(dim=-1)

        # build RDKit Mol
        mol = RWMol()
        for a in atom_preds.tolist():
            mol.AddAtom(Atom(self.idx2elem[a]))
        for (u, v), b in zip(edge_index.t().tolist(), bond_preds.tolist()):
            mol.AddBond(u, v, self.bond_type_map[b])
        SanitizeMol(mol)
        return MolToSmiles(mol)


def build_model(args: Args, fp_loader = None, combinations_names=None):
    # we always want the OptionalInputSPECTRE backbone here
    return MoonshotDiffusion(args, combinations_names, fp_loader)
