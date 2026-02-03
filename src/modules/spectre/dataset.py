import os
import pickle
import torch
import traceback
import sys
from typing import Any, Optional

from itertools import islice
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl

from .args import SPECTREArgs

from ..core.const import DEBUG_LEN, DROP_PERCENTAGE, DATASET_ROOT, NON_SPECTRAL_INPUTS, HSQC_TYPE, \
    C_NMR_TYPE, H_NMR_TYPE, MW_TYPE, MS_TYPE, INPUT_MAP

from ..data.fp_loader import FPLoader
from ..data.inputs import SPECTREInputLoader, MFInputLoader
from ..log import get_logger

logger = get_logger(__file__)


class SPECTREDataset(Dataset):
    def __init__(self, args: SPECTREArgs, fp_loader: FPLoader, split: str = 'train', override_input_types: Optional[list[str]] = None):
        try:
            self.args = args
            self.split = split
            if split != 'train':
                args.requires = args.input_types
            logger.debug(
                f'[MARINADataset] Initializing {split} dataset with input types {args.input_types} and required inputs {args.requires}')
            self.input_types = args.input_types if override_input_types is None else override_input_types
            self.requires = args.requires if override_input_types is None else override_input_types

            with open(os.path.join(DATASET_ROOT, 'index.pkl'), 'rb') as f:
                data: dict[int, Any] = pickle.load(f)
            data = {
                idx: entry for idx, entry in data.items()
                if entry['split'] == split and
                any(
                    entry[f'has_{input_type}']
                    for input_type in self.input_types
                    if input_type not in NON_SPECTRAL_INPUTS
                )
            }
            data_len = len(data)
            logger.debug(
                f'[MARINADataset] Requiring the following items to be present: {self.requires}')
            data = {
                idx: entry for idx, entry in data.items()
                if all(entry[f'has_{dtype}'] for dtype in self.requires)
            }
            logger.debug(
                f'[MARINADataset] Purged {data_len - len(data)}/{data_len} items. {len(data)} items remain')
            logger.debug(f'[MARINADataset] Dataset size: {len(data)}')
            if args.debug and len(data) > DEBUG_LEN:
                logger.debug(
                    f'[MARINADataset] Debug mode activated. Data length set to {DEBUG_LEN}')
                data = dict(islice(data.items(), DEBUG_LEN))

            if len(data) == 0:
                raise RuntimeError(
                    f'[MARINADataset] Dataset split {split} is empty!')

            self.jittering = args.jittering if split == 'train' else 0.0
            self.spectral_loader = SPECTREInputLoader(
                DATASET_ROOT, data, split=split)
            self.mfp_loader = MFInputLoader(fp_loader)

            self.data = list(data.items())

            logger.debug('[MARINADataset] Setup complete!')

        except Exception:
            logger.error(traceback.format_exc())
            logger.error(
                '[MARINADataset] While instantiating the dataset, ran into the above error.')
            sys.exit(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_idx, data_obj = self.data[idx]
        if self.split != 'train':
            input_types = set(self.input_types)
            data_inputs = self.spectral_loader.load(
                data_idx, input_types, jittering=self.jittering)
            mfp = self.mfp_loader.load(data_idx)

            inputs, type_indicator = self._pad_and_stack_input(data_inputs)
            return inputs, mfp, type_indicator
        available_types = {
            'hsqc': data_obj['has_hsqc'],
            'c_nmr': data_obj['has_c_nmr'],
            'h_nmr': data_obj['has_h_nmr'],
            'mass_spec': data_obj['has_mass_spec']
        }
        drop_candidates = [
            k for k, v in available_types.items() if k in self.input_types and v]
        assert len(drop_candidates) > 0, 'Found an empty entry!'

        idx = torch.randint(len(drop_candidates), (1,)).item()
        always_keep = drop_candidates[idx]
        input_types = set(self.input_types)
        for input_type in self.input_types:
            if not data_obj[f'has_{input_type}']:
                input_types.remove(input_type)
            elif (input_type != always_keep and
                  input_type not in self.requires and
                  torch.rand(1).item() < DROP_PERCENTAGE[input_type]):
                input_types.remove(input_type)
        data_inputs = self.spectral_loader.load(
            data_idx, input_types, jittering=self.jittering)
        mfp = self.mfp_loader.load(data_idx)

        inputs, type_indicator = self._pad_and_stack_input(data_inputs)
        return inputs, mfp, type_indicator

    def _pad_and_stack_input(self, data_inputs: dict[str, torch.Tensor]):
        '''
        Type indicators:
        0: HSQC
        1: C NMR
        2: H NMR
        3: MW
        4: Mass Spectrometry
        '''
        inputs = []
        type_indicators = []
        for input_type, input_tensor in data_inputs.items():
            if input_type == 'mw':
                input_tensor = torch.tensor(
                    [input_tensor.item(), 0, 0]).float()
                type_indicator = [MW_TYPE]
            else:
                type_indicator = [INPUT_MAP[input_type]] * len(input_tensor)
            inputs.append(input_tensor)
            type_indicators += type_indicator
        return torch.vstack(inputs), torch.tensor(type_indicators).long()

class SPECTREDataModule(pl.LightningDataModule):
    def __init__(self, args: SPECTREArgs, fp_loader: FPLoader):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.persistent_workers = bool(
            args.persistent_workers and self.num_workers > 0)
        self.fp_loader = fp_loader
        self.test_types = [args.input_types] + args.additional_test_types
        self._fit_is_setup = False
        self._test_is_setup = False

    def setup(self, stage):
        if (stage == "fit" or stage == "validate" or stage is None) and not self._fit_is_setup:
            self.train = SPECTREDataset(
                self.args, self.fp_loader, split='train')
            self.val = [SPECTREDataset(self.args, self.fp_loader, split='val',
                                       override_input_types=input_type) for input_type in self.test_types]
            self._fit_is_setup = True
        if (stage == "test") and not self._test_is_setup:
            self.test = [SPECTREDataset(self.args, self.fp_loader, split='test',
                                        override_input_types=input_type) for input_type in self.test_types]
            self._test_is_setup = True
        if stage == "predict":
            raise NotImplementedError("Predict setup not implemented")

    def __getitem__(self, idx):
        if not self._fit_is_setup:
            self.setup(stage='fit')
        return self.train[idx]

    def train_dataloader(self):
        if not self._fit_is_setup:
            self.setup(stage='fit')
        return DataLoader(
            self.train,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers
        )

    def val_dataloader(self):
        if not self._fit_is_setup:
            self.setup(stage='fit')
        return [DataLoader(
            val_dl,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers
        ) for val_dl in self.val]

    def test_dataloader(self):
        if not self._test_is_setup:
            self.setup(stage='test')
        return [DataLoader(
            test_dl,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers
        ) for test_dl in self.test]

    def _collate_fn(self, batch):
        items = tuple(zip(*batch))
        inputs = pad_sequence([v for v in items[0]], batch_first=True)
        fp = torch.stack(items[1])
        type_indicator = pad_sequence([v for v in items[2]], batch_first=True)
        return inputs, fp, type_indicator
    
    def format_inference_data(self, data: dict[int, Any]) -> dict[str, Any]:
        '''
        Return collated data for inference.
        
        data: stores same thing as input_loader would give:
        {
            'hsqc': ..., # shape: (N, 3)
            'c_nmr': ..., # shape: (N, 1)
            'h_nmr': ..., # shape: (N, 1)
            'mw': ... # shape: (1,)
        }
        
        Usage: 
        >>> inputs = data_module.format_inference_data(data)
        >>> output = model(**inputs)
        '''
        if not self._test_is_setup:
            self.setup(stage='test')
        if 'mw' in data:
            data['mw'] = torch.tensor(data['mw'])
        if 'c_nmr' in data:
            data['c_nmr'] = F.pad(data['c_nmr'].view(-1, 1), (0, 2), "constant", 0)
        if 'h_nmr' in data:
            data['h_nmr'] = F.pad(data['h_nmr'].view(-1, 1), (1, 1), "constant", 0)
        if 'mass_spec' in data:
            data['mass_spec'] = F.pad(data['mass_spec'].view(-1, 1), (0, 1), "constant", 0)
        inputs, type_indicators = self.test[0]._pad_and_stack_input(data)
        inputs, _, type_indicators = self._collate_fn([(inputs, torch.tensor([0.0]), type_indicators)])
        return {'inputs': inputs, 'type_indicator': type_indicators}
        