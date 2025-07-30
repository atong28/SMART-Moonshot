import os
import pickle
import torch
import random
import logging
import traceback
import sys

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
from torch_geometric.data import Batch

from .const import (
    DEBUG_LEN,
    DROP_MW_PERCENTAGE,
    DROP_MS_PERCENTAGE,
    DROP_FORMULA_PERCENTAGE,
    INPUTS_CANONICAL_ORDER,
    ELEM2IDX,
    UNK_IDX,
    FORMULA_RE
)
from .settings import Args

logger = logging.getLogger("lightning")

def init_logger(path):
    logger = logging.getLogger("lightning")
    rank = int(os.environ.get("RANK", 0))
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    if not logger.handlers:
        file_path = os.path.join(path, "logs.txt")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w'): pass
        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(file_path)
        fh.setFormatter(fmt)
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

class MoonshotDataset(Dataset):
    def __init__(self, args: Args, results_path: str, split: str = 'train', overrides: dict | None = None):
        try:
            self.args = args if overrides is None else Args(**{**vars(args), **overrides})
            init_logger(results_path).info(f'[MoonshotDataset] Initializing {split} split')
            self.root = self.args.data_root
            self.split = split
            self.input_types = self.args.input_types
            self.requires = self.args.requires
            self.debug = self.args.debug
            self.jittering = self.args.jittering
            self.use_peak_values = self.args.use_peak_values
            self.elem2idx = ELEM2IDX
            self.unk_idx = UNK_IDX

            with open(os.path.join(self.root, 'index.pkl'), 'rb') as f:
                raw = pickle.load(f)
            # filter by split
            entries = {idx: e for idx, e in raw.items() if e.get('split') == split}
            # filter out entries missing required types
            filtered = {}
            for idx, e in entries.items():
                ok = True
                for dt in self.requires:
                    if dt in ('mw', 'formula'):
                        if e.get(dt) is None:
                            ok = False; break
                    else:
                        if not e.get(f'has_{dt}', False):
                            ok = False; break
                if ok:
                    e['idx'] = idx
                    filtered[idx] = e
            entries = filtered
            if self.debug and len(entries) > DEBUG_LEN:
                entries = dict(list(entries.items())[:DEBUG_LEN])
            if not entries:
                raise RuntimeError(f'No data for split={split}')
            self.entries = list(entries.values())
            init_logger(results_path).info(f'[MoonshotDataset] {len(self.entries)} items loaded')
        except Exception:
            logger.error(traceback.format_exc())
            sys.exit(1)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        data_inputs = {}
        filename = f"{e['idx']}.pt"

        # determine always_keep for dropping
        available = [t for t in ('hsqc','c_nmr','h_nmr','mass_spec')
                     if t in self.input_types and e.get(f'has_{t}', False)]
        always_keep = random.choice(available) if available else None

        # HSQC
        if 'hsqc' in self.input_types and e.get('has_hsqc'):
            h = torch.load(os.path.join(self.root, 'HSQC_NMR', filename), weights_only=True).float()
            if self.jittering and self.split=='train':
                h[:,0] += torch.randn_like(h[:,0]) * self.jittering
                h[:,1] += torch.randn_like(h[:,1]) * self.jittering * 0.1
            if self.use_peak_values:
                h = normalize_hsqc(h)
            data_inputs['hsqc'] = h

        # C_NMR
        if 'c_nmr' in self.input_types and e.get('has_c_nmr'):
            c = torch.load(os.path.join(self.root, 'C_NMR', filename), weights_only=True).float()
            c = c.view(-1,1); c = torch.nn.functional.pad(c, (0,2), 'constant', 0)
            if self.jittering and self.split=='train':
                c += torch.randn_like(c) * self.jittering
            data_inputs['c_nmr'] = c

        # H_NMR
        if 'h_nmr' in self.input_types and e.get('has_h_nmr'):
            h2 = torch.load(os.path.join(self.root, 'H_NMR', filename), weights_only=True).float()
            h2 = h2.view(-1,1); h2 = torch.nn.functional.pad(h2, (1,1), 'constant', 0)
            if self.jittering and self.split=='train':
                h2 += torch.randn_like(h2) * self.jittering * 0.1
            data_inputs['h_nmr'] = h2

        # optionally drop one NMR branch
        if 'c_nmr' in self.input_types and 'h_nmr' in self.input_types:
            if 'c_nmr' not in self.requires and 'h_nmr' not in self.requires and available:
                r = random.random()
                # original probabilities ~0.3984 and 0.2032
                if r <= 0.3984 and always_keep!='c_nmr':
                    data_inputs.pop('c_nmr', None)
                elif r <= 0.3984+0.2032 and always_keep!='h_nmr':
                    data_inputs.pop('h_nmr', None)

        # Mass spec
        if 'mass_spec' in self.input_types and e.get('has_mass_spec'):
            if ('mass_spec' in self.requires) or always_keep=='mass_spec' or random.random() >= DROP_MS_PERCENTAGE:
                m = torch.load(os.path.join(self.root, 'MassSpec', filename), weights_only=True).float()
                m = torch.nn.functional.pad(m, (0,1), 'constant', 0)
                if self.jittering and self.split=='train':
                    noise = torch.zeros_like(m)
                    noise[:,0] = torch.randn_like(m[:,0]) * (m[:,0]/100_000)
                    noise[:,1] = torch.randn_like(m[:,1]) * (m[:,1]/10)
                    m += noise
                data_inputs['mass_spec'] = m

        # MW
        if 'mw' in self.input_types and e.get('mw') is not None:
            if ('mw' in self.requires) or random.random() >= DROP_MW_PERCENTAGE:
                data_inputs['mw'] = torch.tensor(e['mw'], dtype=torch.float)

        # formula idx/cnt + full vector
        if 'formula' in self.input_types and e.get('formula') is not None:
            if ('formula' in self.requires) or random.random() >= DROP_FORMULA_PERCENTAGE:
                counts = parse_formula(e['formula'])
                ordered = []
                if 'C' in counts: ordered.append('C')
                if 'H' in counts: ordered.append('H')
                for elem in sorted(x for x in counts if x not in ('C','H')): ordered.append(elem)
                idxs = [self.elem2idx.get(x, self.unk_idx) for x in ordered]
                cnts = [counts[x] for x in ordered]
                data_inputs['elem_idx'] = torch.tensor(idxs, dtype=torch.long)
                data_inputs['elem_cnt'] = torch.tensor(cnts, dtype=torch.long)
                vec = torch.zeros(len(self.elem2idx)-1, dtype=torch.float)
                for x,cnt in counts.items():
                    j = self.elem2idx.get(x)
                    if j and j>0: vec[j-1] = float(cnt)
                data_inputs['formula_vec'] = vec

        # target graph
        graph = e['graph']  # PyG Data
        return data_inputs, graph


def collate_diffusion(batch):
    dicts, graphs = zip(*batch)
    batch_inputs = {}
    # sequence modalities
    for mod in INPUTS_CANONICAL_ORDER:
        if mod == 'mw': continue
        seqs = [d.get(mod) for d in dicts]
        if all(x is None for x in seqs): continue
        D = next(x.shape[1] for x in seqs if x is not None)
        padded = [(x if x is not None else torch.zeros((0,D))) for x in seqs]
        batch_inputs[mod] = pad_sequence(padded, batch_first=True)
    # mw scalar
    mws = [d.get('mw') for d in dicts]
    if any(x is not None for x in mws):
        batch_inputs['mw'] = torch.tensor([float(x) if x is not None else 0.0 for x in mws])
    # elem_idx/cnt
    idxs = [d.get('elem_idx') for d in dicts]
    if any(x is not None for x in idxs):
        batch_inputs['elem_idx'] = pad_sequence(
            [(x if x is not None else torch.zeros(0,dtype=torch.long)) for x in idxs],
            batch_first=True, padding_value=0)
        cnts = [d.get('elem_cnt') for d in dicts]
        batch_inputs['elem_cnt'] = pad_sequence(
            [(x if x is not None else torch.zeros(0,dtype=torch.long)) for x in cnts],
            batch_first=True, padding_value=0)
    # formula_vec
    vecs = [d.get('formula_vec') for d in dicts]
    if any(v is not None for v in vecs):
        # find a reference tensor for shape
        ref = next(v for v in vecs if v is not None)
        filled = [v if v is not None else torch.zeros_like(ref) for v in vecs]
        batch_inputs['formula_vec'] = torch.stack(filled, dim=0)

    # batch graphs
    graph_batch = Batch.from_data_list(list(graphs))
    return batch_inputs, graph_batch

class MoonshotDataModule(pl.LightningDataModule):
    def __init__(self, args: Args, results_path: str):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.results_path = results_path

    def setup(self, stage=None):
        if stage in ('fit', None):
            self.train = MoonshotDataset(self.args, self.results_path, split='train')
            self.val   = MoonshotDataset(self.args, self.results_path, split='val')
        if stage=='test':
            self.test  = MoonshotDataset(self.args, self.results_path, split='test')

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True,
            collate_fn=collate_diffusion, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False,
            collate_fn=collate_diffusion, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False,
            collate_fn=collate_diffusion, num_workers=self.num_workers)
