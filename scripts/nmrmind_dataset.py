import json
import os
import pickle

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm
import torch
#  TODO: Fix the hardcoded paths
DATASET_ROOT = '/data/nas-gpu/wang/atong/NMRMindDataset'

SPECTRA_TYPES = [
    'HSQC_NMR',
    'H_NMR',
    'C_NMR',
    'COSY_NMR',
    'HMBC_NMR',
]


def ensure_spectra_dirs():
    """Create modality folders under DATASET_ROOT if they don't exist."""
    for t in SPECTRA_TYPES:
        os.makedirs(os.path.join(DATASET_ROOT, t), exist_ok=True)


def strip_c(s: str) -> float:
    # s looks like "C_19.9"
    return float(s.lstrip('C_'))


def strip_h(s: str) -> float:
    # s looks like "H_1.76"
    return float(s.lstrip('H_'))


def save_spectra_tensors(data: dict, idx: int):
    """
    Save spectra for a single molecule as torch.float32 tensors.

    Files:
      - C_NMR/{idx}.pt: 1D tensor of 13C shifts
      - H_NMR/{idx}.pt: 1D tensor of 1H shifts
      - HSQC_NMR/{idx}.pt: 2D tensor [N_peaks, 2] of (H_shift, C_shift) from 'COSY'
      - COSY_NMR/{idx}.pt: 2D tensor [N_peaks, 2] of (H1_shift, H2_shift) from 'HH'
      - HMBC_NMR/{idx}.pt: 2D tensor [N_peaks, 2] of (H_shift, C_shift) from 'HMBC'
    """

    root = DATASET_ROOT

    # 13C_NMR → C_NMR (1D)
    if '13C_NMR' in data and data['13C_NMR']:
        c_shifts = [strip_c(s) for s in data['13C_NMR']]
        c_tensor = torch.tensor(c_shifts, dtype=torch.float32)
        torch.save(c_tensor, os.path.join(root, 'C_NMR', f'{idx}.pt'))

    # 1H_NMR → H_NMR (1D, only chemical shifts)
    # input: [["H_1.76", "11H"], ...]
    if '1H_NMR' in data and data['1H_NMR']:
        h_shifts = [strip_h(pair[0]) for pair in data['1H_NMR']]
        h_tensor = torch.tensor(h_shifts, dtype=torch.float32)
        torch.save(h_tensor, os.path.join(root, 'H_NMR', f'{idx}.pt'))

    # COSY (H–C) → HSQC_NMR (2D)
    # input: [["H_1.49", "C_19.9"], ...]
    if 'COSY' in data and data['COSY']:
        hsqc_rows = []
        for h_str, c_str in data['COSY']:
            hsqc_rows.append([strip_h(h_str), strip_c(c_str)])
        hsqc_tensor = torch.tensor(hsqc_rows, dtype=torch.float32)
        torch.save(hsqc_tensor, os.path.join(root, 'HSQC_NMR', f'{idx}.pt'))

    # HH (H–H COSY) → COSY_NMR (2D)
    # input: [["H_1.49", "H_4.00"], ...]
    if 'HH' in data and data['HH']:
        cosy_rows = []
        for h1_str, h2_str in data['HH']:
            cosy_rows.append([strip_h(h1_str), strip_h(h2_str)])
        cosy_tensor = torch.tensor(cosy_rows, dtype=torch.float32)
        torch.save(cosy_tensor, os.path.join(root, 'COSY_NMR', f'{idx}.pt'))

    # HMBC (H–C) → HMBC_NMR (2D)
    # input: [["H_1.49", "C_111.6"], ...]
    if 'HMBC' in data and data['HMBC']:
        hmbc_rows = []
        for h_str, c_str in data['HMBC']:
            hmbc_rows.append([strip_h(h_str), strip_c(c_str)])
        hmbc_tensor = torch.tensor(hmbc_rows, dtype=torch.float32)
        torch.save(hmbc_tensor, os.path.join(root, 'HMBC_NMR', f'{idx}.pt'))


def parse_dataset(index: dict, split: str) -> dict:
    if split == 'train':
        files = ['train0.json', 'train1.json', 'train2.json', 'train3.json', 'train4.json']
    else:
        files = [f'{split}.json']

    for file in files:
        path = os.path.join(DATASET_ROOT, file)
        with open(path, 'r') as f:
            entries = f.readlines()

        for line in tqdm(entries, desc=f'Reading {file}'):
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)

            mol = Chem.MolFromSmiles(data['smiles'])
            if mol is None:
                continue

            idx = len(index)
            save_spectra_tensors(data, idx)

            index[idx] = {
                'smiles': data['smiles'],
                'has_hsqc': 'COSY' in data, # HSQC corresponds to COSY in the input
                'has_c_nmr': '13C_NMR' in data,
                'has_h_nmr': '1H_NMR' in data,
                'has_cosy': 'HH' in data,
                'has_hmbc': 'HMBC' in data,
                'has_mass_spec': False,
                'formula': data['molecular_formula'],
                'split': split,
                'has_mw': True,
                'mw': rdMolDescriptors.CalcExactMolWt(mol),
            }

    return index

def build_retrieval(index: dict):
    retrieval = {}
    for idx, data in index.items():
        retrieval[idx] = {
            'smiles': data['smiles']
        }
    return retrieval

if __name__ == '__main__':
    ensure_spectra_dirs()

    index = {}
    for split in ('train', 'val', 'test'):
        index = parse_dataset(index, split)

    with open(os.path.join(DATASET_ROOT, 'index.pkl'), 'wb') as f:
        pickle.dump(index, f)
        
    index = pickle.load(open(os.path.join(DATASET_ROOT, 'index.pkl'), 'rb'))
    retrieval = build_retrieval(index)
    with open(os.path.join(DATASET_ROOT, 'retrieval.pkl'), 'wb') as f:
        pickle.dump(retrieval, f)
