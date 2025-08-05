import os
import pickle
from rdkit import Chem
import pandas as pd
from collections import Counter
from tqdm import tqdm

# ─── STEP 1: point to your index.pkl ──────────────────────────────────────────────
# Update this to the real path on your filesystem:
index_path = '/data/nas-gpu/wang/atong/MoonshotDataset/index.pkl'

if not os.path.exists(index_path):
    raise FileNotFoundError(f"Couldn't find index.pkl at {index_path}")

with open(index_path, 'rb') as f:
    data = pickle.load(f)   # data is your dict of {idx: entry}

# ─── STEP 2: count atoms ─────────────────────────────────────────────────────────
atom_counts = Counter()
for entry in tqdm(data.values()):
    smi = entry.get('smiles')
    if not smi:
        continue
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    for atom in mol.GetAtoms():
        atom_counts[ atom.GetSymbol() ] += 1

# ─── STEP 3: tabulate & display ────────────────────────────────────────────────
df = pd.DataFrame(
    atom_counts.items(),
    columns=['atom', 'count']
).sort_values('count', ascending=False).reset_index(drop=True)

print(df)
