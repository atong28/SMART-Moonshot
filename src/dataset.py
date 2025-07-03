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
DROP_ID_PERCENTAGE = 0.5

HSQC_TYPE = 0
C_NMR_TYPE = 1
H_NMR_TYPE = 2
MW_TYPE = 3
ID_TYPE = 4
MS_TYPE = 5

INPUTS_CANONICAL_ORDER = ['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mw', 'iso_dist']

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
            self.cache_path = os.path.join(self.root, f'index_{split}_{self.input_types_encoded}_{self.requires_types_encoded}.pkl')
            if args.use_cached_datasets and os.path.exists(self.cache_path):
                logger.info(f'[MoonshotDataset] Loading filepaths from cache {self.cache_path}')
                with open(self.cache_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(os.path.join(self.root, 'index.pkl'), 'rb') as f:
                    data = pickle.load(f)
                data = {idx: entry for idx, entry in data.items() if entry['split'] == self.split}
                data_len = len(data)
                logger.info(f'[MoonshotDataset] Requiring the following items to be present: {self.requires}')
                data = {idx: entry for idx, entry in data.items() if all(entry[f'has_{dtype}'] for dtype in self.requires)}
                logger.info(f'[MoonshotDataset] Purged {data_len - len(data)}/{data_len} items. {len(data)} items remain')
            
            # if not os.path.exists(self.cache_path) and not args.use_cached_datasets:
            #     logger.info('[MoonshotDataset] Caching processed dataset to workspace')
            #     with open(self.cache_path, 'wb') as f:
            #         pickle.dump(data, f)
            
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
            if self.jittering > 0 and self.split == 'train':
                c_nmr = c_nmr + torch.randn_like(c_nmr) * self.jittering
            data_inputs['c_nmr'] = c_nmr
            
        if 'h_nmr' in self.input_types and data_obj['has_h_nmr']:
            h_nmr = torch.load(os.path.join(self.root, 'H_NMR', filename), weights_only=True).float()
            if self.jittering > 0 and self.split == 'train':
                h_nmr = h_nmr + torch.randn_like(h_nmr) * self.jittering * 0.1
            data_inputs['h_nmr'] = h_nmr
        
        if ('c_nmr' in self.input_types and 'c_nmr' not in self.requires and 
            'h_nmr' in self.input_types and 'h_nmr' not in self.requires):
            random_num = random.random()
            if random_num <= 0.3984:
                c_nmr = torch.tensor([]) 
            elif random_num <= 0.3984 + 0.2032:
                h_nmr = torch.tensor([])

        if 'mass_spec' in self.input_types and data_obj['has_mass_spec']:
            if 'mass_spec' in self.requires or ('mass_spec' not in self.requires and random.random() >= DROP_MS_PERCENTAGE):
                data_inputs['mass_spec'] = torch.load(os.path.join(self.root, 'MassSpec', filename), weights_only=True).float()
            # TODO: should we jitter?

        if 'iso_dist' in self.input_types and data_obj['has_iso_dist']:
            if 'iso_dist' in self.requires or ('iso_dist' not in self.requires and random.random() >= DROP_ID_PERCENTAGE):
                data_inputs['iso_dist'] = torch.load(os.path.join(self.root, 'IsoDist', filename), weights_only=True).float()
            # TODO: should we jitter?

        if 'mw' in self.input_types and data_obj['has_mw']:
            if 'mw' in self.requires or ('mw' not in self.requires and random.random() >= DROP_MW_PERCENTAGE):
                data_inputs['mw'] = torch.tensor([data_obj['mw'], 0, 0]).float() 
        
        inputs, type_indicator = self._pad_and_stack_input(**data_inputs)
        
        if self.fp_type == 'Entropy':
            mfp = self.fp_loader.build_mfp(data_idx)
        elif self.fp_type == 'HYUN':
            mfp = torch.load(os.path.join(self.root, 'HYUN_FP', filename), weights_only=True).float()
        else:
            raise NotImplementedError('MFP type not yet implemented')

        return inputs, mfp, type_indicator
    
    def _pad_and_stack_input(self, hsqc = None, c_nmr = None, h_nmr = None, mass_spec = None, iso_dist = None, mw = None):
        '''
        Type indicators:
        0: HSQC
        1: C NMR
        2: H NMR
        3: MW
        4: Isotopic Distribution
        5: Mass Spectrometry
        '''
        # TODO: can probably remove this assertion at some point
        assert any(input_tensor is not None for input_tensor in (hsqc, c_nmr, h_nmr, mass_spec, iso_dist)), 'Input is nothing!'
        
        inputs = []
        type_indicator = []
        if hsqc is not None:
            inputs.append(hsqc)
            type_indicator += [HSQC_TYPE] * len(hsqc)
        if c_nmr is not None:
            c_nmr = c_nmr.view(-1, 1)
            c_nmr = F.pad(c_nmr, (0, 2), "constant", 0)
            inputs.append(c_nmr)
            type_indicator += [C_NMR_TYPE] * len(c_nmr)
        if h_nmr is not None:
            h_nmr = h_nmr.view(-1, 1)
            h_nmr = F.pad(h_nmr, (1, 1), "constant", 0)
            inputs.append(h_nmr)
            type_indicator += [H_NMR_TYPE] * len(h_nmr)
        if mw is not None:
            inputs.append(mw)
            type_indicator.append(MW_TYPE)
        if iso_dist is not None:
            iso_dist = F.pad(iso_dist, (0, 1), "constant", 0)
            inputs.append(iso_dist)
            type_indicator += [ID_TYPE] * len(iso_dist)
        if mass_spec is not None:
            mass_spec = F.pad(mass_spec, (0, 1), "constant", 0)
            inputs.append(mass_spec)
            type_indicator += [MS_TYPE] * len(mass_spec)
            
        inputs = torch.vstack(inputs)               
        type_indicator = torch.tensor(type_indicator).long()
        return inputs, type_indicator

def collate(batch):
    items = tuple(zip(*batch))
    inputs = pad_sequence([v for v in items[0]], batch_first=True) 
    fp = items[1]
    if type(fp[0][0]) is not str:
        fp = torch.stack(fp)
    if len(items) == 2:
        return (inputs, fp)
    type_indicator = pad_sequence([v for v in items[2]], batch_first=True)
    if len(items) == 3:
        return (inputs, fp, type_indicator)
    raise NotImplementedError("Not implemented")

class MoonshotDataModule(pl.LightningDataModule):
    def __init__(self, args: Args, results_path: str, fp_loader: EntropyFPLoader):
        super().__init__()
        if len(args.requires) == 0 or (len(args.requires) == 1 and args.requires[0] == 'mw'):
            raise ValueError('You must require at least one datatype, and it cannot be molecular weight!')
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
                self.val = MoonshotDataset(self.args, self.results_path, self.fp_loader, split='val')
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
                self.test = MoonshotDataset(self.args, self.results_path, self.fp_loader, split='test')
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
