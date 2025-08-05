from diffms.utils import to_dense
from diffms.diffusion.distributions import DistributionNodes
from dataset import MoonshotDataModule
from const import ATOM_DECODER, VALENCY, ATOM_TO_WEIGHT

class AbstractDatasetInfos:
    def complete_infos(self, n_nodes, node_types):
        self.input_dims = None
        self.output_dims = None
        self.num_classes = len(node_types)
        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)

    def compute_input_output_dims(self, datamodule: MoonshotDataModule, extra_features, domain_features):
        _, example_batch = next(iter(datamodule.train_dataloader()))
        ex_dense, node_mask = to_dense(example_batch.x, example_batch.edge_index, example_batch.edge_attr,
                                             example_batch.batch)
        example_data = {'X_t': ex_dense.X, 'E_t': ex_dense.E, 'y_t': example_batch['y'], 'node_mask': node_mask}

        self.input_dims = {'X': example_batch['x'].size(1),
                           'E': example_batch['edge_attr'].size(1),
                           'y': example_batch['y'].size(1) + datamodule.args.dim_model + 1}      # + 1 due to time conditioning

        ex_extra_feat = extra_features(example_data)
        self.input_dims['X'] += ex_extra_feat.X.size(-1)
        self.input_dims['E'] += ex_extra_feat.E.size(-1)
        self.input_dims['y'] += ex_extra_feat.y.size(-1)

        ex_extra_molecular_feat = domain_features(example_data)
        self.input_dims['X'] += ex_extra_molecular_feat.X.size(-1)
        self.input_dims['E'] += ex_extra_molecular_feat.E.size(-1)
        self.input_dims['y'] += ex_extra_molecular_feat.y.size(-1)

        self.output_dims = {'X': example_batch['x'].size(1),
                            'E': example_batch['edge_attr'].size(1),
                            'y': example_batch['y'].size(1)}

class MoonshotInfos(AbstractDatasetInfos):
    def __init__(self, datamodule: MoonshotDataModule):
        self.name = 'MoonshotDataset'
        self.input_dims = None
        self.output_dims = None
        self.remove_h = False

        self.atom_decoder = ATOM_DECODER
        self.atom_encoder = {atom: i for i, atom in enumerate(self.atom_decoder)}
        self.atom_weights = {i: ATOM_TO_WEIGHT.get(atom, 0) for i, atom in enumerate(self.atom_decoder)}
        self.valencies = VALENCY
        self.num_atom_types = len(self.atom_decoder)
        self.max_weight = max(self.atom_weights.values())
        self.n_nodes = datamodule.node_counts()
        
        self.max_n_nodes = len(self.n_nodes) - 1
        self.node_types = datamodule.node_types()
        self.edge_types = datamodule.edge_counts()
        valencies = datamodule.valency_count(self.max_n_nodes)
        self.valency_distribution = valencies

        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)