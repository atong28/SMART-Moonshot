import os
import pickle
import torch
import traceback
import sys
import logging
import random
from itertools import islice
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
from torch_geometric.data import Batch
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.rdchem import BondType as BT

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

from const import (
    DEBUG_LEN, DROP_MW_PERCENTAGE, DROP_MS_PERCENTAGE, DROP_FORMULA_PERCENTAGE,
    INPUTS_CANONICAL_ORDER, ELEM2IDX, UNK_IDX, FORMULA_RE, FILTER_ATOMS
)
from settings import SpectreArgs, MoonshotArgs

torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])

def is_main_process():
    return int(os.environ.get("RANK", 0)) == 0

def init_logger(path):
    logger = logging.getLogger("lightning")
    if is_main_process():
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
    if not logger.handlers:
        file_path = os.path.join(path, "logs.txt")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as fp:
            pass

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(file_path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger

def parse_formula(formula: str) -> dict[str,int]:
    """
    Turn "C20H25BrN2O2" → {"C":20, "H":25, "Br":1, "N":2, "O":2}
    """
    counts: dict[str,int] = {}
    for elem, cnt in FORMULA_RE.findall(formula):
        counts[elem] = int(cnt) if cnt else 1
    return counts

def normalize_hsqc(hsqc):
    """
    Normalizes each column of the input HSQC to have zero mean and unit standard deviation.
    Parameters:
    hsqc (torch.Tensor): Input tensor of shape (n, 3).
    Returns:
    torch.Tensor: Normalized hsqc of shape (n, 3).
    """    
    
    assert(len(hsqc.shape)==2 and hsqc.shape[1]==3)
    '''normalize only peak intensities, and separate positive and negative peaks'''
    selected_values = hsqc[hsqc[:,2] > 0, 2]
    # do min_max normalization with in the range of 0.5 to 1.5
    if len(selected_values) > 1:
        min_pos = selected_values.min()
        max_pos = selected_values.max()
        if min_pos == min_pos:
            hsqc[hsqc[:,2]>0,2] = 1
        else:
            hsqc[hsqc[:,2]>0,2] = (selected_values - min_pos) / (max_pos - min_pos) + 0.5
    elif len(selected_values) == 1:
        hsqc[hsqc[:,2]>0,2] = 1
    
    # do min_max normalization with in the range of -0.5 to -1.5
    selected_values = hsqc[hsqc[:,2] < 0, 2]
    if len(selected_values) > 1:
        min_neg = selected_values.min()
        max_neg = selected_values.max()
        if min_neg == max_neg:
            hsqc[hsqc[:,2]<0,2] = -1
        else:
            hsqc[hsqc[:,2]<0,2] = (min_neg - selected_values ) / (max_neg - min_neg) - 0.5
    elif len(selected_values) == 1:
        hsqc[hsqc[:,2]<0,2] = -1

    return hsqc

def filter_with_atom_types(smi):
    try:
        mol = Chem.MolFromSmiles(smi)

        if "." in smi:
            return False
        
        if Descriptors.MolWt(mol) >= 1500:
            return False
        
        for atom in mol.GetAtoms():
            if atom.GetFormalCharge() != 0:
                return False
            if atom.GetSymbol() not in FILTER_ATOMS:
                return False
    except:
        return False
    
    return True
class MoonshotDataset(Dataset):
    def __init__(self, args: SpectreArgs, results_path: str, split: str = 'train', overrides: dict | None = None,):
        '''
        Assertions:
        c_nmr and h_nmr are either both present in input_types or both not present.
        '''
        try:
            logger = init_logger(results_path)
            if overrides is not None:
                args = SpectreArgs(**{**vars(args), **overrides})
            logger.info(f'[MoonshotDataset] Initializing {split} dataset with input types {args.input_types} and required inputs {args.requires}')
            self.root = args.data_root
            self.split = split
            self.input_types = args.input_types
            self.requires = args.requires
            self.fp_type = args.fp_type
            self.debug = args.debug
            self.input_types_encoded = sum(2 ** i for i, input_type in enumerate(INPUTS_CANONICAL_ORDER) if input_type in self.input_types)
            self.requires_types_encoded = sum(2 ** i for i, input_type in enumerate(INPUTS_CANONICAL_ORDER) if input_type in self.requires)

            with open(os.path.join(self.root, 'index.pkl'), 'rb') as f:
                data = pickle.load(f)
            data = {
                idx: entry for idx, entry in data.items()
                if entry['split'] == self.split and
                any(
                    entry[f'has_{input_type}']
                    for input_type in self.input_types
                    if input_type not in ('mw', 'formula')
                )
            }
            logger.info(f'[MoonshotDataset] Loaded initial candidates, {len(data)} entries.')
            logger.info(f'[MoonshotDataset] Requiring the following items to be present: {self.requires}')
            data_len = len(data)
            data = {
                idx: entry for idx, entry in tqdm(data.items(), desc='Filtering, stage 1')
                if all(
                    entry[f'has_{dtype}']
                    if dtype not in ('mw', 'formula')
                    else entry[dtype] is not None
                    for dtype in self.requires
                )
            }
            data = {
                idx: entry for idx, entry in tqdm(data.items(), desc='Filtering, stage 2')
                if os.path.exists(os.path.join(args.data_root, 'Graph', f'{idx}.pt')) and filter_with_atom_types(entry['smiles'])
            }
            logger.info(f'[MoonshotDataset] Purged {data_len - len(data)}/{data_len} items. {len(data)} items remain')
            
            if self.debug and len(data) > DEBUG_LEN:
                logger.info(f'[MoonshotDataset] Debug mode activated. Data length set to {DEBUG_LEN}')
                data = dict(islice(data.items(), DEBUG_LEN))
            self.data = list(data.items())
            if len(self.data) == 0:
                raise RuntimeError(f'[MoonshotDataset] Dataset split {split} is empty!')
            self.jittering = args.jittering
            self.use_peak_values = args.use_peak_values
            self.elem2idx = ELEM2IDX
            self.unk_idx   = UNK_IDX
            self.idx2elem = {i: e for e, i in self.elem2idx.items()}
            logger.info('[MoonshotDataset] Setup complete!')

        except Exception:
            logger.error(traceback.format_exc())
            logger.error('[MoonshotDataset] While instantiating the dataset, ran into the above error. It is likely that your dataset is not formatted properly.')
            sys.exit(1)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_idx, data_obj = self.data[idx]
        filename = f'{data_idx}.pt'
        available_types = {
            'hsqc': data_obj['has_hsqc'],
            'c_nmr': data_obj['has_c_nmr'],
            'h_nmr': data_obj['has_h_nmr'],
            'mass_spec': data_obj['has_mass_spec']
        }
        available_types = [k for k, v in available_types.items() if k in self.input_types and v]
        always_keep = random.choice(available_types)
        data_inputs = {}
        if 'hsqc' in self.input_types and data_obj['has_hsqc']:
            hsqc = torch.load(os.path.join(self.root, 'HSQC_NMR', filename), weights_only=True).float()
            if self.jittering > 0 and self.split == 'train':
                hsqc[:,0] = hsqc[:,0] + torch.randn_like(hsqc[:,0]) * self.jittering
                hsqc[:,1] = hsqc[:,1] + torch.randn_like(hsqc[:,1]) * self.jittering * 0.1
            if self.use_peak_values:
                hsqc = normalize_hsqc(hsqc)
            data_inputs['hsqc'] = hsqc

        if 'c_nmr' in self.input_types and data_obj['has_c_nmr']:
            c_nmr = torch.load(os.path.join(self.root, 'C_NMR', filename), weights_only=True).float()
            c_nmr = c_nmr.view(-1,1)                   # (N,1)
            c_nmr = F.pad(c_nmr, (0,2), "constant", 0) # -> (N,3)
            if self.jittering > 0 and self.split == 'train':
                c_nmr = c_nmr + torch.randn_like(c_nmr) * self.jittering
            data_inputs['c_nmr'] = c_nmr
            
        if 'h_nmr' in self.input_types and data_obj['has_h_nmr']:
            h_nmr = torch.load(os.path.join(self.root, 'H_NMR', filename), weights_only=True).float()
            h_nmr = h_nmr.view(-1,1)                    # (N,1)
            h_nmr = F.pad(h_nmr, (1,1), "constant", 0)  # -> (N,3)
            if self.jittering > 0 and self.split == 'train':
                h_nmr = h_nmr + torch.randn_like(h_nmr) * self.jittering * 0.1
            data_inputs['h_nmr'] = h_nmr
        
        # optional drop of one NMR branch only if HSQC is present already
        if ('c_nmr' in self.input_types and 'c_nmr' not in self.requires and
            'h_nmr' in self.input_types and 'h_nmr' not in self.requires):
            r = random.random()
            if r <= 0.3984 and 'c_nmr' != always_keep:
                # drop C-NMR
                data_inputs.pop('c_nmr', None)
            elif r <= 0.3984 + 0.2032 and 'h_nmr' != always_keep:
                # drop H-NMR
                data_inputs.pop('h_nmr', None)

        if 'mass_spec' in self.input_types and data_obj['has_mass_spec']:
            if (
                'mass_spec' in self.requires 
                or 'mass_spec' == always_keep 
                or ('mass_spec' not in self.requires and random.random() >= DROP_MS_PERCENTAGE)
            ):
                ms = torch.load(os.path.join(self.root, 'MassSpec', filename), weights_only=True).float()
                ms = F.pad(ms, (0,1), "constant", 0)
                if self.jittering > 0 and self.split == 'train':
                    noise = torch.zeros_like(ms)
                    noise[:, 0] = torch.randn_like(ms[:, 0]) * ms[:, 0] / 100_000  # jitter m/z
                    noise[:, 1] = torch.randn_like(ms[:, 1]) * ms[:, 1] / 10
                    ms = ms + noise
                data_inputs['mass_spec'] = ms

        assert len(data_inputs) != 0, f'Always keep was {always_keep} and data has input types {[data_obj[f"has_{input_type}"] for input_type in self.input_types if input_type != "mw"]} and data inputs is {data_inputs}'

        if 'mw' in self.input_types and data_obj['has_mw']:
            if 'mw' in self.requires or ('mw' not in self.requires and random.random() >= DROP_MW_PERCENTAGE):
                data_inputs['mw'] = torch.tensor(data_obj['mw'], dtype=torch.float)
        
        if 'formula' in self.input_types and data_obj.get('formula', None) is not None:
            if 'formula' in self.requires or ('formula' not in self.requires and random.random() >= DROP_FORMULA_PERCENTAGE):
                formula = data_obj['formula']
                elem_counts = parse_formula(formula)
                # Hill order: C, H, then alphabetical
                ordered = []
                if 'C' in elem_counts: ordered.append('C')
                if 'H' in elem_counts: ordered.append('H')
                for e in sorted(e for e in elem_counts if e not in ('C','H')):
                    ordered.append(e)
                # map to indices & counts
                idxs = [ self.elem2idx.get(e, UNK_IDX) for e in ordered ]
                cnts = [ elem_counts[e]       for e in ordered ]
                # store as 1D tensors
                data_inputs['elem_idx'] = torch.tensor(idxs, dtype=torch.long)
                data_inputs['elem_cnt'] = torch.tensor(cnts, dtype=torch.long)
        
        graph_path = os.path.join(self.root, 'Graph', filename)
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"No graph file for smiles {data_obj['smiles']} at {graph_path}")
        
        graph = torch.load(graph_path, weights_only=True)
        return data_inputs, graph

def collate(batch):
    dicts, graphs = zip(*batch)
    batch_inputs = {}

    # 1) Handle all the *sequence* modalities
    for mod in INPUTS_CANONICAL_ORDER:
        if mod == "mw":
            # skip MW here—handle below
            continue

        seqs = [d.get(mod) for d in dicts]
        # if none of the samples have this modality, skip it entirely
        if all(x is None for x in seqs):
            continue

        # replace missing with empty (0×D) tensors
        # find the first real tensor to infer D
        D = next(x.shape[1] for x in seqs if isinstance(x, torch.Tensor) and x.ndim == 2)
        seqs = [
            x if (isinstance(x, torch.Tensor) and x.ndim == 2) else torch.zeros((0, D), dtype=torch.float)
            for x in seqs
        ]
        # now pad them into a (B, L_mod, D) tensor
        batch_inputs[mod] = pad_sequence(seqs, batch_first=True)

    # 2) Handle MW *scalar* specially
    mw_vals = [d.get("mw") for d in dicts]
    if any(v is not None for v in mw_vals):
        # replace None with 0.0 (or another sentinel if you like)
        mw_floats = [float(v) if v is not None else 0.0 for v in mw_vals]
        # create a (B,) tensor of scalars
        batch_inputs["mw"] = torch.tensor(mw_floats, dtype=torch.float)

    # 3) Handle element‐group tokens (idx + count)
    elem_idx_seqs = [d.get('elem_idx') for d in dicts]
    if any(x is not None for x in elem_idx_seqs):
        # pad element‐ID sequences (pad_value=0)
        batch_inputs['elem_idx'] = pad_sequence(
            [x if x is not None else torch.zeros(0,dtype=torch.long)
             for x in elem_idx_seqs],
            batch_first=True,
            padding_value=0
        )
        # pad count sequences (pad_value=0)
        cnt_seqs = [d.get('elem_cnt') for d in dicts]
        batch_inputs['elem_cnt'] = pad_sequence(
            [x if x is not None else torch.zeros(0,dtype=torch.long)
             for x in cnt_seqs],
            batch_first=True,
            padding_value=0
        )

    batched_graph = Batch.from_data_list(graphs)
    return batch_inputs, batched_graph

class MoonshotDataModule(pl.LightningDataModule):
    def __init__(self, args: SpectreArgs, moonshot_args: MoonshotArgs, results_path: str):
        super().__init__()
        self.args = args
        self.batch_size = moonshot_args.batch_size
        self.num_workers = moonshot_args.num_workers
        self.collate_fn = collate
        self.persistent_workers = self.args.persistent_workers
        self.results_path = results_path
        
        self._fit_is_setup = False
        self._test_is_setup = False
    
    def setup(self, stage):
        if (stage == "fit" or stage == "validate" or stage is None) and not self._fit_is_setup:
            self.train = MoonshotDataset(self.args, self.results_path, split='train')
            self.val = MoonshotDataset(self.args, self.results_path, split='val', overrides={'requires': self.args.input_types})
            self._fit_is_setup = True
        if (stage == "test") and not self._test_is_setup:
            self.test = MoonshotDataset(self.args, self.results_path, split='test', overrides={'requires': self.args.input_types})
            self._test_is_setup = True
        if stage == "predict":
            raise NotImplementedError("Predict setup not implemented")
    
    def __getitem__(self, idx):
        if not self._fit_is_setup:
            self.setup(stage = 'fit')
        return self.train[idx]
    
    def train_dataloader(self):
        if not self._fit_is_setup:
            self.setup(stage = 'fit')
        return DataLoader(
            self.train,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True, 
            persistent_workers=self.persistent_workers
        )

    def val_dataloader(self):
        if not self._fit_is_setup:
            self.setup(stage = 'fit')
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn, 
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers
        )
    def test_dataloader(self):
        if not self._test_is_setup:
            self.setup(stage = 'test')
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn, 
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers
        )

    def valency_count(self, max_n_nodes):
        valencies = torch.zeros(3 * max_n_nodes - 2)   # Max valency possible if everything is connected

        # No bond, single bond, double bond, triple bond, aromatic bond
        multiplier = torch.tensor([0, 1, 2, 3, 1.5])

        for _, data in self.train_dataloader():
            n = data.x.shape[0]

            for atom in range(n):
                edges = data.edge_attr[data.edge_index[0] == atom]
                edges_total = edges.sum(dim=0)
                valency = (edges_total * multiplier).sum()
                valencies[valency.long().item()] += 1
        valencies = valencies / valencies.sum()
        return valencies
    
    def node_counts(self, max_nodes_possible=150):
        all_counts = torch.zeros(max_nodes_possible)
        for desc, loader in [('train', self.train_dataloader()), ('val', self.val_dataloader()), ('test', self.test_dataloader())]:
            for _, data in tqdm(loader, desc=f'Computing node counts in {desc} loader'):
                unique, counts = torch.unique(data.batch, return_counts=True)
                for count in counts:
                    all_counts[count] += 1
        max_index = max(all_counts.nonzero())
        all_counts = all_counts[:max_index + 1]
        all_counts = all_counts / all_counts.sum()
        return all_counts

    def node_types(self):
        num_classes = None
        for (_, data) in self.train_dataloader():
            num_classes = data.x.shape[1]
            break

        counts = torch.zeros(num_classes)

        for i, (_, data) in enumerate(self.train_dataloader()):
            counts += data.x.sum(dim=0)

        counts = counts / counts.sum()
        return counts

    def edge_counts(self):
        num_classes = None
        for (_, data) in tqdm(self.train_dataloader(), desc='Computing node types'):
            num_classes = data.edge_attr.shape[1]
            break

        d = torch.zeros(num_classes, dtype=torch.float)

        for i, (_, data) in tqdm(enumerate(self.train_dataloader()), desc='Computing edge counts'):
            unique, counts = torch.unique(data.batch, return_counts=True)

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