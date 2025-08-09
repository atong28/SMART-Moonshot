import pickle
import os
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch_geometric.data import Data
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.rdchem import BondType as BT

ATOM_DECODER = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H']
BONDS = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
ATOM_TYPES = {atom: i for i, atom in enumerate(ATOM_DECODER)}

def process(idx, smiles):
    """
    Process a single molecule.
    """
    
    RDLogger.DisableLog('rdApp.*')
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f'Failed to get molecule from smiles {smiles}')
        N = mol.GetNumAtoms()
        type_idx = []
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol not in ATOM_TYPES:
                raise ValueError(f'Unknown atom type {symbol} encountered while processing {smiles}')
            type_idx.append(ATOM_TYPES[symbol])
        row, col, edge_type = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_type += 2 * [BONDS[bond.GetBondType()] + 1]
        if len(row) == 0:
            raise ValueError(f'No bonds found in molecule {smiles}')
        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        edge_attr = F.one_hot(edge_type, num_classes=len(BONDS) + 1).to(torch.float)
        perm = (edge_index[0] * N + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_attr = edge_attr[perm]
        x = F.one_hot(torch.tensor(type_idx), num_classes=len(ATOM_TYPES)).float()
        y = torch.zeros((1, 0), dtype=torch.float)
        inchi_canonical = Chem.MolToInchi(mol)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=idx, inchi=inchi_canonical)
        return data
    except Exception as e:
        print(e)
        return None

DATA_ROOT = '/data/nas-gpu/wang/atong/MoonshotDataset'

index = pickle.load(open(os.path.join(DATA_ROOT, 'index.pkl'), 'rb'))

count = 0
for idx, entry in tqdm(index.items()):
    graph = process(idx, entry['smiles'])
    if graph is None:
        count += 1
        continue
    torch.save(graph, os.path.join(DATA_ROOT, 'Graph', f'{idx}.pt'))

print(f'Finished processing, skipped {count} molecules out of {len(index)}')