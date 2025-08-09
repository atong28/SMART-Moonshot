import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, softmax

# === Constants follow MoleMCL conventions ===
# atom types: 0..118 are real; 119 is the [MASK] token
num_atom_type = 120
num_chirality_tag = 3
# bond types: 0..3 real; 4=self-loop; 5=[MASK] token
num_bond_type = 6
num_bond_direction = 3


# --------------------
# Convs (from MoleMCL)
# --------------------
class GINConv(MessagePassing):
    def __init__(self, emb_dim, out_dim, aggr="add", **kwargs):
        kwargs.setdefault("aggr", aggr)
        super().__init__(**kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, out_dim),
        )
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.zeros(x.size(0), 2, device=edge_attr.device, dtype=edge_attr.dtype)
        self_loop_attr[:, 0] = 4  # self-loop bond type
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        e = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        return self.propagate(edge_index, x=x, edge_attr=e)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super().__init__()
        self.emb_dim = emb_dim
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.zeros(x.size(0), 2, device=edge_attr.device, dtype=edge_attr.dtype)
        self_loop_attr[:, 0] = 4
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        e = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        norm = self.norm(edge_index, x.size(0), x.dtype)
        x = self.linear(x)
        return self.propagate(edge_index, x=x, edge_attr=e, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add"):
        super().__init__()
        self.aggr = aggr
        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope
        self.weight_linear = nn.Linear(emb_dim, heads * emb_dim)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))
        self.bias = nn.Parameter(torch.Tensor(emb_dim))
        self.edge_embedding1 = nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, heads * emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.zeros(x.size(0), 2, device=edge_attr.device, dtype=edge_attr.dtype)
        self_loop_attr[:, 0] = 4
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        e = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        return self.propagate(edge_index, x=x, edge_attr=e)

    def message(self, edge_index, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j = x_j + edge_attr
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])
        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias
        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="mean"):
        super().__init__()
        self.emb_dim = emb_dim
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.zeros(x.size(0), 2, device=edge_attr.device, dtype=edge_attr.dtype)
        self_loop_attr[:, 0] = 4
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        e = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        x = self.linear(x)
        return self.propagate(edge_index, x=x, edge_attr=e)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)


# --------------------
# GNN encoder (MoleMCL)
# --------------------
class GNN(nn.Module):
    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0.0, gnn_type="gin"):
        super().__init__()
        if num_layer < 2:
            raise ValueError("Number of GNN layers must be > 1")
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.gnns = nn.ModuleList()
        for _ in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))
            else:
                raise ValueError(f"Unknown gnn_type={gnn_type}")

        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(num_layer)])

    def forward(self, x, edge_index, edge_attr):
        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        if self.JK == "concat":
            node_rep = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_rep = h_list[-1]
        elif self.JK == "max":
            node_rep = torch.max(torch.stack(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            node_rep = torch.sum(torch.stack(h_list, dim=0), dim=0)[0]
        else:
            raise ValueError(f"Unknown JK={self.JK}")

        return node_rep


# --------------------
# GraphCL head (pool + MLP)
# --------------------
class GraphCLHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.pool = global_mean_pool
        self.projection = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True), nn.Linear(in_dim, in_dim))

    def forward(self, node_rep, batch_ids):
        g = self.pool(node_rep, batch_ids)
        z = self.projection(g)
        return g, z  # graph embedding (pre-proj), and projection


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    # z1, z2: (B, D)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    sim = torch.matmul(z1, z2.t()) / temperature  # (B,B)
    # positives on diagonal; negatives off-diagonal
    pos = torch.diag(sim)
    # denominator: sum over each row except the diagonal
    exp_sim = torch.exp(sim)
    denom = exp_sim.sum(dim=1) - torch.exp(pos)
    loss = -torch.log(torch.exp(pos) / (denom + 1e-8)).mean()
    return loss


# ==========================
# Lightning Module (MoleMCL)
# ==========================
class MoleMCLModule(pl.LightningModule):
    def __init__(
        self,
        num_layer: int = 5,
        emb_dim: int = 300,
        gnn_type: str = "gin",
        JK: str = "last",
        drop_ratio: float = 0.0,
        mask_edge: bool = True,
        alpha: float = 1.0,           # weight for contrastive loss
        temperature: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        node_class_weights: Optional[torch.Tensor] = None,
        edge_class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["node_class_weights", "edge_class_weights"])

        self.gnn = GNN(num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio, gnn_type=gnn_type)
        self.head = GraphCLHead(emb_dim)
        # reconstruction heads
        self.linear_pred_atoms = nn.Linear(emb_dim, 120)  # MoleMCL uses 512 atom classes
        self.linear_pred_bonds = nn.Linear(emb_dim, 4)    # 4 bond types for CE

        self.ce = nn.CrossEntropyLoss()
        self.mask_edge = mask_edge
        self.alpha = alpha
        self.temperature = temperature
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.mask_edge = mask_edge
        self.alpha = alpha
        self.temperature = temperature
        self.lr = lr
        self.weight_decay = weight_decay

        # placeholders; will be set in on_fit_start
        self._node_w = node_class_weights
        self._edge_w = edge_class_weights
        self._label_smoothing = label_smoothing
        self.ce_nodes = None
        self.ce_edges = None

    # ----- helpers -----
    @staticmethod
    def _node_acc(logits: torch.Tensor, labels: torch.Tensor) -> float:
        if labels.numel() == 0:
            return 0.0
        pred = logits.argmax(dim=-1)
        return (pred == labels).float().mean().item()

    def _restore_unmasked_view(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rebuild x and edge_attr with original labels at masked positions.
        Assumes paired directed edges; reverse index = idx + 1.
        """
        x0 = batch.x.clone()
        if hasattr(batch, "masked_atom_indices") and batch.masked_atom_indices.numel() > 0:
            x0[batch.masked_atom_indices] = batch.mask_node_label

        e0 = batch.edge_attr.clone()
        if self.mask_edge and hasattr(batch, "connected_edge_indices") and batch.connected_edge_indices.numel() > 0:
            eidx = batch.connected_edge_indices
            e0[eidx] = batch.mask_edge_label
            # restore reverse edges (paired ordering)
            eidx_rev = eidx + 1
            valid = eidx_rev < e0.size(0)
            e0[eidx_rev[valid]] = batch.mask_edge_label[valid]
        return x0, e0

    # ----- core forward -----
    def encode(self, x, edge_index, edge_attr):
        return self.gnn(x, edge_index, edge_attr)

    def forward(self, batch):
        node_rep = self.encode(batch.x, batch.edge_index, batch.edge_attr)
        _, z = self.head(node_rep, batch.batch)
        return z

    # ----- one step (shared by train/val/test) -----
    def _step(self, batch, stage: str):
        # masked view (already in batch.x/edge_attr)
        node_rep_masked = self.encode(batch.x, batch.edge_index, batch.edge_attr)
        _, z_masked = self.head(node_rep_masked, batch.batch)

        # unmasked/original view (restore labels into a copy)
        x0, e0 = self._restore_unmasked_view(batch)
        node_rep_orig = self.encode(x0, batch.edge_index, e0)
        _, z_orig = self.head(node_rep_orig, batch.batch)

        # --- reconstruction: atoms ---
        loss_node = torch.tensor(0.0, device=batch.x.device)
        acc_node = 0.0
        if hasattr(batch, "masked_atom_indices") and batch.masked_atom_indices.numel() > 0:
            logits_nodes = self.linear_pred_atoms(node_rep_masked[batch.masked_atom_indices])
            labels_nodes = batch.mask_node_label[:, 0].long()
            loss_node = self.ce_nodes(logits_nodes, labels_nodes)
            acc_node = self._node_acc(logits_nodes, labels_nodes)

        # --- reconstruction: edges (optional) ---
        loss_edge = torch.tensor(0.0, device=batch.x.device)
        acc_edge = 0.0
        if self.mask_edge and hasattr(batch, "connected_edge_indices") and batch.connected_edge_indices.numel() > 0:
            eidx = batch.connected_edge_indices
            u = batch.edge_index[0, eidx]
            v = batch.edge_index[1, eidx]
            edge_rep = node_rep_masked[u] + node_rep_masked[v]
            logits_edges = self.linear_pred_bonds(edge_rep)
            labels_edges = batch.mask_edge_label[:, 0].long()
            loss_edge = self.ce_edges(logits_edges, labels_edges)
            acc_edge = self._node_acc(logits_edges, labels_edges)

        # --- contrastive (masked vs original) ---
        loss_cl = nt_xent(z_orig, z_masked, temperature=self.temperature)

        # total
        loss = loss_node + loss_edge + self.alpha * loss_cl

        # logging
        self.log_dict(
            {
                f"{stage}/loss": loss,
                f"{stage}/loss_node": loss_node,
                f"{stage}/loss_edge": loss_edge,
                f"{stage}/loss_cl": loss_cl,
                f"{stage}/acc_node": acc_node,
                f"{stage}/acc_edge": acc_edge,
            },
            prog_bar=(stage == "train"),
            on_step=(stage == "train"),
            on_epoch=True,
            batch_size=batch.num_graphs,
        )
        return loss

    # ----- Lightning hooks -----
    
    def on_fit_start(self):
        # If weights weren’t passed, try to pull from datamodule
        if self._node_w is None or self._edge_w is None:
            dm = getattr(self.trainer, "datamodule", None)
            if dm is not None and hasattr(dm, "class_weights"):
                node_w, edge_w = dm.class_weights
                if self._node_w is None and node_w is not None:
                    self._node_w = node_w.to(self.device)
                if self._edge_w is None and edge_w is not None:
                    self._edge_w = edge_w.to(self.device)

        # Build CE criteria with weights (fall back to unweighted if still None)
        self.ce_nodes = nn.CrossEntropyLoss(
            weight=self._node_w, label_smoothing=self._label_smoothing
        )
        self.ce_edges = nn.CrossEntropyLoss(
            weight=self._edge_w, label_smoothing=self._label_smoothing
        )
    
    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        # ---- normal forward (so val losses still get logged) ----
        loss = self._step(batch, "val")

        # ---- DEBUG: check for prediction collapse & class imbalance ----
        with torch.no_grad():
            # Rebuild masked node reps and logits (same as in _step)
            node_rep_masked = self.encode(batch.x, batch.edge_index, batch.edge_attr)

            # Nodes
            if hasattr(batch, "masked_atom_indices") and batch.masked_atom_indices.numel() > 0:
                node_logits = self.linear_pred_atoms(node_rep_masked[batch.masked_atom_indices])
                node_labels = batch.mask_node_label[:, 0].long()

                # label distribution and majority baseline
                num_node_classes = node_logits.size(-1)
                node_label_counts = torch.bincount(node_labels, minlength=num_node_classes).float()
                node_label_dist = node_label_counts / node_label_counts.sum().clamp_min(1)
                node_majority = (node_label_counts.max() / node_label_counts.sum().clamp_min(1)).item()

                # predicted distribution
                node_pred = node_logits.argmax(dim=-1)
                node_pred_counts = torch.bincount(node_pred, minlength=num_node_classes).float()
                node_pred_dist = node_pred_counts / node_pred_counts.sum().clamp_min(1)
                node_acc = (node_pred == node_labels).float().mean().item()
                node_top5 = node_logits.topk(5, dim=-1).indices.eq(node_labels[:, None]).any(dim=1).float().mean().item()
            else:
                node_majority = node_acc = node_top5 = 0.0
                node_label_dist = torch.tensor([], device=self.device)
                node_pred_dist = torch.tensor([], device=self.device)

            # Edges
            if self.mask_edge and hasattr(batch, "connected_edge_indices") and batch.connected_edge_indices.numel() > 0:
                eidx = batch.connected_edge_indices
                u = batch.edge_index[0, eidx]
                v = batch.edge_index[1, eidx]
                edge_rep = node_rep_masked[u] + node_rep_masked[v]
                edge_logits = self.linear_pred_bonds(edge_rep)
                edge_labels = batch.mask_edge_label[:, 0].long()

                num_edge_classes = edge_logits.size(-1)
                edge_label_counts = torch.bincount(edge_labels, minlength=num_edge_classes).float()
                edge_label_dist = edge_label_counts / edge_label_counts.sum().clamp_min(1)
                edge_majority = (edge_label_counts.max() / edge_label_counts.sum().clamp_min(1)).item()

                edge_pred = edge_logits.argmax(dim=-1)
                edge_pred_counts = torch.bincount(edge_pred, minlength=num_edge_classes).float()
                edge_pred_dist = edge_pred_counts / edge_pred_counts.sum().clamp_min(1)
                edge_acc = (edge_pred == edge_labels).float().mean().item()
            else:
                edge_majority = edge_acc = 0.0
                edge_label_dist = torch.tensor([], device=self.device)
                edge_pred_dist = torch.tensor([], device=self.device)

            # Log once per epoch (first batch) to avoid spam
            if batch_idx == 0:
                self.log_dict(
                    {
                        "debug/node_majority_baseline": node_majority,
                        "debug/node_acc": node_acc,
                        "debug/node_top5": node_top5,
                        "debug/edge_majority_baseline": edge_majority,
                        "debug/edge_acc": edge_acc,
                    },
                    prog_bar=False, on_step=False, on_epoch=True, batch_size=batch.num_graphs
                )
                # Also log distributions as histograms if your logger supports it
                if node_label_dist.numel() > 0:
                    self.logger.experiment.log({"debug/node_label_dist": node_label_dist.detach().cpu().numpy()})
                    self.logger.experiment.log({"debug/node_pred_dist": node_pred_dist.detach().cpu().numpy()})
                if edge_label_dist.numel() > 0:
                    self.logger.experiment.log({"debug/edge_label_dist": edge_label_dist.detach().cpu().numpy()})
                    self.logger.experiment.log({"debug/edge_pred_dist": edge_pred_dist.detach().cpu().numpy()})

        return loss


    '''def validation_step(self, batch, batch_idx):
        # edges
        self._step(batch, "val")'''

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return opt
