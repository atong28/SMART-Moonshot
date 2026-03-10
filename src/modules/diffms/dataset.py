import os
import pickle
from itertools import islice
from typing import Optional, Any
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import pytorch_lightning as pl
from tqdm import tqdm
from ..core.const import DATASET_ROOT, DEBUG_LEN
from ..data.inputs import GraphInputLoader, RDKitMFInputLoader
from ..log import get_logger
from .args import DiffMSArgs
from .distributions import DistributionNodes
from .utils import to_dense
from .extra_features import ExtraFeatures, ExtraMolecularFeatures
from .visualizations import MolecularVisualization

logger = get_logger(__file__)


@dataclass
class DiffMSDatasetInfo:
    n_nodes: torch.Tensor
    max_n_nodes: int
    node_types: torch.Tensor
    edge_types: torch.Tensor
    valency_dist: torch.Tensor
    nodes_dist: DistributionNodes
    input_dims: dict[str, int]
    output_dims: dict[str, int]

class DiffMSDataset(Dataset):
    def __init__(self, args: DiffMSArgs, split: str = 'train'):
        self.args = args
        self.split = split
        with open(os.path.join(DATASET_ROOT, 'index.pkl'), 'rb') as f:
            data_dict: dict[int, Any] = pickle.load(f)
        self.data_dict = {
            idx: entry for idx, entry in data_dict.items()
            if entry['split'] == split
        }
        self.data_dict = dict(enumerate(self.data_dict.values()))
        if args.debug and len(self.data_dict) > DEBUG_LEN:
            self.data_dict = dict(islice(self.data_dict.items(), DEBUG_LEN))
        # Prefer Arrow-backed graphs when available; fall back to SMILES otherwise.
        self.graph_input_loader = GraphInputLoader(
            self.data_dict,
            root=DATASET_ROOT,
            split=split,
            use_arrow=None,
        )
        self.mf_input_loader = RDKitMFInputLoader(self.data_dict)

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        x, edge_index, edge_attr, smiles = self.graph_input_loader.load(idx)
        y = self.mf_input_loader.load(idx)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, smiles=smiles)

class DiffMSDataModule(pl.LightningDataModule):
    def __init__(self, args: DiffMSArgs):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.persistent_workers = bool(args.persistent_workers and self.num_workers > 0)
        self._fit_is_setup = False
        self._test_is_setup = False

    def setup(self, stage: Optional[str]):
        if stage == 'fit':
            self.train = DiffMSDataset(self.args, split='train')
            self.val = DiffMSDataset(self.args, split='val')
            self._fit_is_setup = True
        elif stage == 'test':
            self.test = DiffMSDataset(self.args, split='test')
            self._test_is_setup = True
    
    def train_dataloader(self):
        if not self._fit_is_setup:
            self.setup(stage='fit')
        return DataLoader(
            self.train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        if not self._fit_is_setup:
            self.setup(stage='fit')
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers,
        )
    
    def test_dataloader(self):
        if not self._test_is_setup:
            self.setup(stage='test')
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers,
        )

    def node_counts(self, max_nodes_possible=150):
        all_counts = torch.zeros(max_nodes_possible)
        for loader in [self.train_dataloader(), self.val_dataloader()]:
            for data in tqdm(loader, desc='Calculating node counts'):
                _, counts = torch.unique(data.batch, return_counts=True)
                for count in counts:
                    all_counts[count] += 1
        max_index = max(all_counts.nonzero())
        all_counts = all_counts[:max_index + 1]
        all_counts = all_counts / all_counts.sum()
        return all_counts

    def node_types(self) -> torch.Tensor:
        num_classes = next(iter(self.train_dataloader())).x.shape[1]
        counts = torch.zeros(num_classes)
        for data in tqdm(self.train_dataloader(), desc='Calculating node count'):
            counts += data.x.sum(dim=0)
        counts = counts / counts.sum()
        return counts

    def edge_counts(self) -> torch.Tensor:
        num_classes = next(iter(self.train_dataloader())).edge_attr.shape[1]
        d = torch.zeros(num_classes, dtype=torch.float)
        for data in tqdm(self.train_dataloader(), desc='Calculating edge count'):
            _, counts = torch.unique(data.batch, return_counts=True)

            all_pairs = 0
            for count in counts:
                all_pairs += count * (count - 1)

            num_edges = data.edge_index.shape[1]
            num_non_edges = all_pairs - num_edges

            edge_types = data.edge_attr.sum(dim=0)
            assert num_non_edges >= 0
            d[0] += num_non_edges
            d[1:] += edge_types[1:]

        d = d / d.sum()
        return d

    def valency_count(self, max_n_nodes):
        valencies = torch.zeros(3 * max_n_nodes - 2)
        multiplier = torch.tensor([0, 1, 2, 3, 1.5])

        for data in tqdm(self.train_dataloader(), desc='Calculating valency count'):
            n = data.x.shape[0]

            for atom in range(n):
                edges = data.edge_attr[data.edge_index[0] == atom]
                edges_total = edges.sum(dim=0)
                valency = (edges_total * multiplier).sum()
                valencies[valency.long().item()] += 1
        valencies = valencies / valencies.sum()
        return valencies
    
    def get_infos_and_features(self, recompute: bool = False) -> tuple[DiffMSDatasetInfo, MolecularVisualization, ExtraFeatures, ExtraMolecularFeatures]:
        info = {}
        if os.path.exists(os.path.join(DATASET_ROOT, 'dataset_info.pkl')) and not recompute:
            with open(os.path.join(DATASET_ROOT, 'dataset_info.pkl'), 'rb') as f:
                info = pickle.load(f)
        else:
            n_nodes = self.node_counts()
            max_n_nodes = len(n_nodes) - 1
            info = {
                'n_nodes': n_nodes,
                'max_n_nodes': max_n_nodes,
                'node_types': self.node_types(),
                'edge_types': self.edge_counts(),
                'valency_dist': self.valency_count(max_n_nodes),
                'nodes_dist': DistributionNodes(n_nodes)
            }
            with open(os.path.join(DATASET_ROOT, 'dataset_info.pkl'), 'wb') as f:
                pickle.dump(info, f)
        extra_features = ExtraFeatures(max_n_nodes=info['max_n_nodes'])
        domain_features = ExtraMolecularFeatures()
        info['input_dims'], info['output_dims'] = self.compute_input_output_dims(extra_features, domain_features)
        visualization_tools = MolecularVisualization()
        return DiffMSDatasetInfo(**info), visualization_tools, extra_features, domain_features
    
    def compute_input_output_dims(self, extra_features, domain_features):
        ex_batch: Data = next(iter(self.train_dataloader()))
        ex_dense, node_mask = to_dense(ex_batch.x, ex_batch.edge_index, ex_batch.edge_attr, ex_batch.batch)
        example_data = {'X_t': ex_dense.X, 'E_t': ex_dense.E, 'y_t': ex_batch['y'], 'node_mask': node_mask}

        input_dims = {
            'X': ex_batch['x'].size(1),
            'E': ex_batch['edge_attr'].size(1),
            'y': ex_batch['y'].size(1) + 1
        }      # + 1 due to time conditioning

        ex_extra_feat = extra_features(example_data)
        input_dims['X'] += ex_extra_feat.X.size(-1)
        input_dims['E'] += ex_extra_feat.E.size(-1)
        input_dims['y'] += ex_extra_feat.y.size(-1)

        ex_extra_molecular_feat = domain_features(example_data)
        input_dims['X'] += ex_extra_molecular_feat.X.size(-1)
        input_dims['E'] += ex_extra_molecular_feat.E.size(-1)
        input_dims['y'] += ex_extra_molecular_feat.y.size(-1)

        output_dims = {
            'X': ex_batch['x'].size(1),
            'E': ex_batch['edge_attr'].size(1),
            'y': ex_batch['y'].size(1)
        }
        return input_dims, output_dims

