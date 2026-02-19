import os
import pickle
import torch
import traceback
import sys
from itertools import islice
from typing import Any, List, Optional

from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl

from .args import MARINAArgs

from ..core.const import DEBUG_LEN, DROP_PERCENTAGE, INPUTS_CANONICAL_ORDER, DATASET_ROOT, NON_SPECTRAL_INPUTS

from ..data.fp_loader import FPLoader
from ..data.inputs import MARINAInputLoader, MFInputLoader
from ..log import get_logger

logger = get_logger(__file__)


class MARINADataset(Dataset):
    def __init__(self, args: MARINAArgs, fp_loader: FPLoader, split: str = 'train', override_input_types: Optional[list[str]] = None):
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
            self.spectral_loader = MARINAInputLoader(
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
            return self.spectral_loader.load(data_idx, input_types), self.mfp_loader.load(data_idx)
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
        return self.spectral_loader.load(data_idx, input_types, jittering=self.jittering), self.mfp_loader.load(data_idx)


class MARINADataModule(pl.LightningDataModule):
    def __init__(self, args: MARINAArgs, fp_loader: FPLoader):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.persistent_workers = bool(
            args.persistent_workers and self.num_workers > 0)
        self.fp_loader = fp_loader
        self.test_types = [args.input_types] + args.additional_test_types
        self.test_types = [types for types in self.test_types if all(t in args.input_types for t in types)]
        self._fit_is_setup = False
        self._test_is_setup = False

    def setup(self, stage: Optional[str]):
        if (stage == "fit" or stage == "validate" or stage is None) and not self._fit_is_setup:
            self.train = MARINADataset(
                self.args,
                self.fp_loader,
                split='train'
            )

            self.val = [
                MARINADataset(
                    self.args,
                    self.fp_loader,
                    split='val',
                    override_input_types=input_type
                ) for input_type in self.test_types
            ]

            self._fit_is_setup = True

        if (stage == "test") and not self._test_is_setup:
            self.test = [
                MARINADataset(
                    self.args,
                    self.fp_loader,
                    split='test',
                    override_input_types=input_type
                ) for input_type in self.test_types
            ]

            self._test_is_setup = True

        if stage == "predict":
            raise NotImplementedError("Predict setup not implemented")

    def __getitem__(self, idx):
        if not self._fit_is_setup:
            self.setup(stage='fit')
        return self.train[idx]

    def train_dataloader(self) -> DataLoader:
        """_summary_

        Returns:
            _type_: _description_
        """
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

    def val_dataloader(self) -> List[DataLoader]:
        """_summary_

        Returns:
            _type_: _description_
        """
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

    def test_dataloader(self) -> List[DataLoader]:
        """_summary_

        Returns:
            _type_: _description_
        """
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
        """
        batch: list of (data_inputs: dict, mfp: Tensor)
        returns: (batch_inputs: dict[str→Tensor], batch_fps: Tensor)
        """
        dicts, fps = zip(*batch)
        batch_inputs = {}
        for mod in INPUTS_CANONICAL_ORDER:
            seqs = [d.get(mod) for d in dicts]
            if all(x is None for x in seqs):
                continue
            D = next(x.shape[1] for x in seqs if isinstance(
                x, torch.Tensor) and x.ndim == 2)
            seqs = [
                x if (isinstance(x, torch.Tensor) and x.ndim ==
                      2) else torch.zeros((0, D), dtype=torch.float)
                for x in seqs
            ]
            batch_inputs[mod] = pad_sequence(seqs, batch_first=True)

        batch_fps = torch.stack(fps, dim=0)
        return batch_inputs, batch_fps

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
        if 'mw' in data:
            data['mw'] = torch.tensor(data['mw']).view(1, 1)
        batch_inputs, _ = self._collate_fn([(data, torch.tensor([0.0]))])
        return {'batch': batch_inputs}