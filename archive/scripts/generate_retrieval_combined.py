import pickle
import os

OLD_DATA_ROOT = '/data/nas-gpu/wang/atong/MoonshotDatasetv3'
NEW_DATA_ROOT = '/data/nas-gpu/wang/atong/CombinedDataset'

with open(os.path.join(OLD_DATA_ROOT, 'retrieval.pkl'), 'rb') as f:
    retrieval = pickle.load(f)
with open(os.path.join(NEW_DATA_ROOT, 'index.pkl'), 'rb') as f:
    new_index = pickle.load(f)
old_smiles = {entry['smiles'] for entry in retrieval.values()}

for idx, entry in new_index.items():
    if entry['smiles'] not in old_smiles:
        retrieval[len(retrieval)] = {'smiles': entry['smiles']}
        old_smiles.add(entry['smiles'])

with open(os.path.join(NEW_DATA_ROOT, 'retrieval.pkl'), 'wb') as f:
    pickle.dump(retrieval, f)