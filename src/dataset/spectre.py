import os
import pickle
import torch
import traceback
import sys
import logging
import random
from itertools import islice
from typing import Any

from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl

from ..spectre.const import DEBUG_LEN, DROP_PERCENTAGE, INPUTS_CANONICAL_ORDER, DATASET_ROOT
from ..spectre.settings import SPECTREArgs
from ..spectre.fp_loader import FPLoader
from .inputs import SpectralInputLoader, MFInputLoader

logger = logging.getLogger('lightning')

class SPECTREDataset(Dataset):
    def __init__(self, args: SPECTREArgs, fp_loader: FPLoader, split: str = 'train'):
        try:
            if split != 'train':
                args.requires = args.input_types
            logger.info(f'[SPECTREDataset] Initializing {split} dataset with input types {args.input_types} and required inputs {args.requires}')
            self.input_types = args.input_types
            self.requires = args.requires

            with open(os.path.join(DATASET_ROOT, 'index.pkl'), 'rb') as f:
                data: dict[int, Any] = pickle.load(f)
            data = {
                idx: entry for idx, entry in data.items()
                if entry['split'] == split and
                any(
                    entry[f'has_{input_type}']
                    for input_type in self.input_types
                    if input_type not in ('mw', 'formula')
                )
            }
            data_len = len(data)
            logger.info(f'[SPECTREDataset] Requiring the following items to be present: {self.requires}')
            data = {
                idx: entry for idx, entry in data.items()
                if all(entry[f'has_{dtype}'] for dtype in self.requires)
            }
            logger.info(f'[SPECTREDataset] Purged {data_len - len(data)}/{data_len} items. {len(data)} items remain')
            
            if args.debug and len(data) > DEBUG_LEN:
                logger.info(f'[SPECTREDataset] Debug mode activated. Data length set to {DEBUG_LEN}')
                data = dict(islice(data.items(), DEBUG_LEN))

            if len(data) == 0:
                raise RuntimeError(f'[SPECTREDataset] Dataset split {split} is empty!')
            
            self.jittering = args.jittering
            self.spectral_loader = SpectralInputLoader(DATASET_ROOT, data)
            self.mfp_loader = MFInputLoader(fp_loader)
            self.data = list(data.items())
            logger.info('[SPECTREDataset] Setup complete!')
        
        except Exception:
            logger.error(traceback.format_exc())
            logger.error('[SPECTREDataset] While instantiating the dataset, ran into the above error.')
            sys.exit(1)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_idx, data_obj = self.data[idx]
        available_types = {
            'hsqc': data_obj['has_hsqc'],
            'c_nmr': data_obj['has_c_nmr'],
            'h_nmr': data_obj['has_h_nmr'],
            'mass_spec': data_obj['has_mass_spec']
        }
        drop_candidates = [k for k, v in available_types.items() if k in self.input_types and v]
        assert len(drop_candidates) > 0, 'Found an empty entry!'
        
        always_keep = random.choice(drop_candidates)
        input_types = set(self.input_types)
        for input_type in self.input_types:
            if not data_obj[f'has_{input_type}']:
                input_types.remove(input_type)
            elif (input_type != always_keep and 
                  input_type not in self.requires and 
                  random.random() < DROP_PERCENTAGE[input_type]):
                input_types.remove(input_type)
        return self.spectral_loader.load(data_idx, input_types, jittering = self.jittering), self.mfp_loader.load(data_idx)

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

    # 4) Stack your fingerprints
    batch_fps = torch.stack(fps, dim=0)
    return batch_inputs, batch_fps

class SPECTREDataModule(pl.LightningDataModule):
    def __init__(self, args: SPECTREArgs, fp_loader: FPLoader):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.collate_fn = collate
        self.persistent_workers = bool(args.persistent_workers and self.num_workers > 0)
        self.fp_loader = fp_loader
        
        self._fit_is_setup = False
        self._test_is_setup = False
    
    def setup(self, stage):
        if (stage == "fit" or stage == "validate" or stage is None) and not self._fit_is_setup:
            self.train = SPECTREDataset(self.args, self.fp_loader, split='train')
            self.val = SPECTREDataset(self.args, self.fp_loader, split='val')
            self._fit_is_setup = True
        if (stage == "test") and not self._test_is_setup:
            self.test = SPECTREDataset(self.args, self.fp_loader, split='test')
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