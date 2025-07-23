import os
import pickle
import torch
import traceback
import sys
import logging
import random

from itertools import islice, combinations

from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl

from .settings import Args
from .fp_loaders.entropy import EntropyFPLoader


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

DEBUG_LEN = 3000

DROP_MW_PERCENTAGE = 0.5
DROP_MS_PERCENTAGE = 0.5

HSQC_TYPE = 0
C_NMR_TYPE = 1
H_NMR_TYPE = 2
MW_TYPE = 3
ID_TYPE = 4
MS_TYPE = 5

INPUTS_CANONICAL_ORDER = ['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mw']

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
    def __init__(self, args: Args, results_path: str, fp_loader: EntropyFPLoader, split: str = 'train', overrides: dict | None = None,):
        '''
        Assertions:
        c_nmr and h_nmr are either both present in input_types or both not present.
        '''
        try:
            logger = init_logger(results_path)
            if overrides is not None:
                args = Args(**{**vars(args), **overrides})
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
            data = {idx: entry for idx, entry in data.items() if entry['split'] == self.split and any(entry[f'has_{input_type}'] for input_type in self.input_types if input_type != 'mw')}
            data_len = len(data)
            logger.info(f'[MoonshotDataset] Requiring the following items to be present: {self.requires}')
            data = {idx: entry for idx, entry in data.items() if all(entry[f'has_{dtype}'] for dtype in self.requires)}
            logger.info(f'[MoonshotDataset] Purged {data_len - len(data)}/{data_len} items. {len(data)} items remain')
            
            if self.debug and len(data) > DEBUG_LEN:
                logger.info(f'[MoonshotDataset] Debug mode activated. Data length set to {DEBUG_LEN}')
                data = dict(islice(data.items(), DEBUG_LEN))
            self.data = list(data.items())
            if len(self.data) == 0:
                raise RuntimeError(f'[MoonshotDataset] Dataset split {split} is empty!')
            self.jittering = args.jittering
            self.use_peak_values = args.use_peak_values
            self.fp_loader = fp_loader
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
                
        if self.fp_type == 'Entropy':
            mfp = self.fp_loader.build_mfp(data_idx)
        elif self.fp_type == 'HYUN':
            mfp = torch.load(os.path.join(self.root, 'HYUN_FP', filename), weights_only=True).float()
        else:
            raise NotImplementedError()
        
        return data_inputs, mfp

def collate(batch):
    """
    batch: list of (data_inputs: dict, mfp: Tensor)
    returns: (batch_inputs: dict[str→Tensor], batch_fps: Tensor)
    """
    dicts, fps = zip(*batch)
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

    # 3) Stack your fingerprints
    batch_fps = torch.stack(fps, dim=0)
    return batch_inputs, batch_fps

class MoonshotDataModule(pl.LightningDataModule):
    def __init__(self, args: Args, results_path: str, fp_loader: EntropyFPLoader):
        super().__init__()
        self.args = args
        self.batch_size = self.args.batch_size
        self.num_workers = self.args.num_workers
        self.collate_fn = collate
        self.persistent_workers = self.args.persistent_workers
        self.combinations_list, self.combinations_names = self._get_combinations()
        self.validate_all = args.validate_all
        self.results_path = results_path
        self.fp_loader = fp_loader
    
    def _get_combinations(self):
        required = set(self.args.requires)
        all_inputs = set(self.args.input_types)
        optional = all_inputs - required
        combinations_list = []
        for r in range(len(optional) + 1):
            for subset in combinations(optional, r):
                combo = required.union(subset)
                combinations_list.append([
                    input_type for input_type in INPUTS_CANONICAL_ORDER if input_type in combo
                ])
        combinations_names = ['+'.join(combo) if len(combo) < len(self.args.input_types) else 'ALL' for combo in combinations_list]
        zipped = sorted(zip(combinations_names, combinations_list))
        combinations_names, combinations_list = zip(*zipped)
        combinations_names = list(combinations_names)
        combinations_list = list(combinations_list)
        return combinations_list, combinations_names
    
    def setup(self, stage):
        if stage == "fit" or stage == "validate" or stage is None:
            self.train = MoonshotDataset(self.args, self.results_path, self.fp_loader, split='train')
            if self.args.validate_all:
                self.val = [
                    MoonshotDataset(
                        self.args,
                        self.results_path,
                        self.fp_loader,
                        split='val',
                        overrides={'requires': combo, 'input_types': combo}
                    ) for combo in self.combinations_list
                ]
            else:
                self.val = MoonshotDataset(self.args, self.results_path, self.fp_loader, split='val', overrides={'requires': self.args.input_types})
        if stage == "test":
            if self.args.validate_all:
                self.test = [
                    MoonshotDataset(
                        self.args,
                        self.results_path,
                        self.fp_loader,
                        split='test',
                        overrides={'requires': combo, 'input_types': combo}
                    ) for combo in self.combinations_list
                ]
            else:
                self.test = MoonshotDataset(self.args, self.results_path, self.fp_loader, split='test', overrides={'requires': self.args.input_types})
        if stage == "predict":
            raise NotImplementedError("Predict setup not implemented")
    
    def train_dataloader(self):
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
        if self.validate_all:
            return [
                DataLoader(
                    val_dl,
                    batch_size=self.batch_size,
                    collate_fn=self.collate_fn, 
                    num_workers=self.num_workers,
                    pin_memory=True,
                    persistent_workers=self.persistent_workers
                )
                for val_dl in self.val
            ]
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn, 
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers
        )
    def test_dataloader(self):
        if self.validate_all:
            return [
                DataLoader(
                    test_dl,
                    batch_size=self.batch_size,
                    collate_fn=self.collate_fn, 
                    num_workers=self.num_workers,
                    pin_memory=True,
                    persistent_workers=self.persistent_workers
                )
                for test_dl in self.test
            ]
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn, 
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers
        )
