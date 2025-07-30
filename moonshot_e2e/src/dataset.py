# moonshot_e2e/src/dataset.py

import os
import pickle
import torch
import traceback
import sys
import logging
import random
import re
from itertools import islice, combinations

from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl

from torch_geometric.data import Batch   # **NEW**

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
    # your existing normalization…
    selected_values = hsqc[hsqc[:,2] > 0, 2]
    if len(selected_values) > 1:
        min_pos = selected_values.min()
        max_pos = selected_values.max()
        if min_pos == max_pos:
            hsqc[hsqc[:,2]>0,2] = 1
        else:
            hsqc[hsqc[:,2]>0,2] = (selected_values - min_pos) / (max_pos - min_pos) + 0.5
    elif len(selected_values) == 1:
        hsqc[hsqc[:,2]>0,2] = 1

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
    def __init__(self,
                 args: Args,
                 results_path: str,
                 fp_loader = None,  # kept for signature compatibility
                 split: str = 'train',
                 overrides: dict | None = None):
        '''
        Assertions:
        c_nmr and h_nmr are either both present in input_types or both not present.
        '''
        try:
            logger = init_logger(results_path)
            if overrides is not None:
                args = Args(**{**vars(args), **overrides})
            logger.info(f'[MoonshotDataset] Initializing {split} dataset '
                        f'with input types {args.input_types} and requires {args.requires}')
            self.root = args.data_root
            self.split = split
            self.input_types = args.input_types
            self.requires    = args.requires
            self.debug       = args.debug

            # load index; **must** already have `entry["graph"] = Data(...)`
            with open(os.path.join(self.root, 'index.pkl'), 'rb') as f:
                raw = pickle.load(f)

            # filter by split and required modalities
            data = {
                idx: e for idx,e in raw.items()
                if e.get('split') == split
            }
            def has_all(e):
                for dt in self.requires:
                    if dt in ('mw','formula'):
                        if e.get(dt) is None: return False
                    else:
                        if not e.get(f'has_{dt}', False): return False
                return True
            data = {i:e for i,e in data.items() if has_all(e)}

            if self.debug and len(data) > DEBUG_LEN:
                data = dict(islice(data.items(), DEBUG_LEN))

            if not data:
                raise RuntimeError(f'[MoonshotDataset] No data for split={split}')

            self.entries = list(data.items())
            self.jittering      = args.jittering
            self.use_peak_values= args.use_peak_values
            self.elem2idx       = ELEM2IDX
            self.unk_idx        = UNK_IDX

            logger.info('[MoonshotDataset] Setup complete!')

        except Exception:
            logger.error(traceback.format_exc())
            sys.exit(1)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        data_idx, e = self.entries[idx]
        filename = f'{data_idx}.pt'
        data_inputs = {}

        # --- your existing modality‐loading logic, verbatim ---
        # HSQC
        if 'hsqc' in self.input_types and e.get('has_hsqc'):
            hsqc = torch.load(os.path.join(self.root, 'HSQC_NMR', filename),
                              weights_only=True).float()
            if self.jittering > 0 and self.split == 'train':
                hsqc[:,0] += torch.randn_like(hsqc[:,0]) * self.jittering
                hsqc[:,1] += torch.randn_like(hsqc[:,1]) * (self.jittering*0.1)
            if self.use_peak_values:
                hsqc = normalize_hsqc(hsqc)
            data_inputs['hsqc'] = hsqc

        # C_NMR
        if 'c_nmr' in self.input_types and e.get('has_c_nmr'):
            c = torch.load(os.path.join(self.root, 'C_NMR', filename),
                           weights_only=True).float()
            c = c.view(-1,1); c = F.pad(c, (0,2), 'constant', 0)
            if self.jittering > 0 and self.split=='train':
                c += torch.randn_like(c) * self.jittering
            data_inputs['c_nmr'] = c

        # H_NMR
        if 'h_nmr' in self.input_types and e.get('has_h_nmr'):
            h2 = torch.load(os.path.join(self.root, 'H_NMR', filename),
                            weights_only=True).float()
            h2 = h2.view(-1,1); h2 = F.pad(h2, (1,1), 'constant', 0)
            if self.jittering > 0 and self.split=='train':
                h2 += torch.randn_like(h2) * (self.jittering*0.1)
            data_inputs['h_nmr'] = h2

        # optionally drop one branch…
        if ('c_nmr' in self.input_types and 'h_nmr' in self.input_types
            and 'c_nmr' not in self.requires and 'h_nmr' not in self.requires):
            r = random.random()
            if r <= 0.3984:
                data_inputs.pop('c_nmr', None)
            elif r <= 0.3984+0.2032:
                data_inputs.pop('h_nmr', None)

        # Mass spec
        if 'mass_spec' in self.input_types and e.get('has_mass_spec'):
            if ('mass_spec' in self.requires
                or random.random() >= DROP_MS_PERCENTAGE):
                m = torch.load(os.path.join(self.root, 'MassSpec', filename),
                               weights_only=True).float()
                m = F.pad(m, (0,1), 'constant', 0)
                if self.jittering > 0 and self.split=='train':
                    noise = torch.zeros_like(m)
                    noise[:,0] = torch.randn_like(m[:,0])*(m[:,0]/100_000)
                    noise[:,1] = torch.randn_like(m[:,1])*(m[:,1]/10)
                    m += noise
                data_inputs['mass_spec'] = m

        # MW
        if 'mw' in self.input_types and e.get('mw') is not None:
            if ('mw' in self.requires
                or random.random() >= DROP_MW_PERCENTAGE):
                data_inputs['mw'] = torch.tensor(e['mw'], dtype=torch.float)

        # formula as idx+cnt
        if 'formula' in self.input_types and e.get('formula') is not None:
            if ('formula' in self.requires
                or random.random() >= DROP_FORMULA_PERCENTAGE):
                counts = parse_formula(e['formula'])
                ordered = []
                if 'C' in counts: ordered.append('C')
                if 'H' in counts: ordered.append('H')
                for elm in sorted(x for x in counts if x not in ('C','H')):
                    ordered.append(elm)
                idxs = [self.elem2idx.get(x, self.unk_idx) for x in ordered]
                cnts = [counts[x] for x in ordered]
                data_inputs['elem_idx'] = torch.tensor(idxs, dtype=torch.long)
                data_inputs['elem_cnt'] = torch.tensor(cnts, dtype=torch.long)

        # -------------------------------------------------------------------
        # **MODIFIED**: instead of building a fingerprint, pull in the pre‐built graph
        # (you must have done that upfront when you created index.pkl)
        graph = e['graph']
        # stash smiles on the Data object for sampling/metrics
        graph.smiles = e.get('smiles', None)
        return data_inputs, graph

# -----------------------------------------------------------------------------
# **NEW** collate that pads your modalities AND batches PyG graphs

def collate(batch):
    dicts, graphs = zip(*batch)
    batch_inputs = {}

    # 1) sequence modalities
    for mod in INPUTS_CANONICAL_ORDER:
        if mod == 'mw': continue
        seqs = [d.get(mod) for d in dicts]
        if all(x is None for x in seqs):
            continue
        # infer D
        D = next(x.shape[1] for x in seqs if isinstance(x, torch.Tensor) and x.ndim==2)
        padded = [
            x if (isinstance(x,torch.Tensor) and x.ndim==2)
            else torch.zeros((0,D), dtype=torch.float)
            for x in seqs
        ]
        batch_inputs[mod] = pad_sequence(padded, batch_first=True)

    # 2) MW scalar
    mws = [d.get('mw') for d in dicts]
    if any(x is not None for x in mws):
        vals = [float(x) if x is not None else 0.0 for x in mws]
        batch_inputs['mw'] = torch.tensor(vals, dtype=torch.float)

    # 3) element tokens
    idxs = [d.get('elem_idx') for d in dicts]
    if any(x is not None for x in idxs):
        batch_inputs['elem_idx'] = pad_sequence(
            [ x if x is not None else torch.zeros(0,dtype=torch.long) for x in idxs ],
            batch_first=True, padding_value=0
        )
        cnts = [d.get('elem_cnt') for d in dicts]
        batch_inputs['elem_cnt'] = pad_sequence(
            [ x if x is not None else torch.zeros(0,dtype=torch.long) for x in cnts ],
            batch_first=True, padding_value=0
        )

    # 4) batch the graphs
    graph_batch = Batch.from_data_list(graphs)

    return batch_inputs, graph_batch

class MoonshotDataModule(pl.LightningDataModule):
    def __init__(self, args: Args, results_path: str, fp_loader = None):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.persistent_workers = args.persistent_workers
        self.results_path = results_path

        # build combinations logic as before…
        from itertools import combinations
        optional = set(args.input_types) - set(args.requires)
        combos = []
        for r in range(len(optional)+1):
            for subset in combinations(optional, r):
                combo = set(args.requires) | set(subset)
                combos.append([m for m in INPUTS_CANONICAL_ORDER if m in combo])
        names = ['+'.join(c) if len(c)<len(args.input_types) else 'ALL' for c in combos]
        zipped = sorted(zip(names, combos))
        self.combinations_names, self.combinations_list = zip(*zipped)

    def setup(self, stage=None):
        if stage in (None, 'fit', 'validate'):
            self.train = MoonshotDataset(self.args, self.results_path, None, split='train')
            if self.args.validate_all:
                self.val = [
                    MoonshotDataset(self.args, self.results_path, None,
                                    split='val',
                                    overrides={'input_types':combo,'requires':combo})
                    for combo in self.combinations_list
                ]
            else:
                self.val = MoonshotDataset(self.args, self.results_path, None,
                                           split='val',
                                           overrides={'input_types':self.args.input_types,
                                                      'requires':self.args.input_types})

        if stage in (None, 'test'):
            if self.args.validate_all:
                self.test = [
                    MoonshotDataset(self.args, self.results_path, None,
                                    split='test',
                                    overrides={'input_types':combo,'requires':combo})
                    for combo in self.combinations_list
                ]
            else:
                self.test = MoonshotDataset(self.args, self.results_path, None,
                                            split='test',
                                            overrides={'input_types':self.args.input_types,
                                                       'requires':self.args.input_types})

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers,
            collate_fn=collate,
        )

    def val_dataloader(self):
        if self.args.validate_all:
            return [
                DataLoader(ds,
                           batch_size=self.batch_size,
                           shuffle=False,
                           num_workers=self.num_workers,
                           pin_memory=True,
                           persistent_workers=self.persistent_workers,
                           collate_fn=collate)
                for ds in self.val
            ]
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers,
            collate_fn=collate,
        )

    def test_dataloader(self):
        if self.args.validate_all:
            return [
                DataLoader(ds,
                           batch_size=1, # override for single sample
                           shuffle=False,
                           num_workers=self.num_workers,
                           pin_memory=True,
                           persistent_workers=self.persistent_workers,
                           collate_fn=collate)
                for ds in self.test
            ]
        return DataLoader(
            self.test,
            batch_size=1, # override for single sample
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers,
            collate_fn=collate,
        )
