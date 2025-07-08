import pickle
import random

random.seed(0)

with open('/workspace/CombinedDataset/index.pkl', 'rb') as f:
    data = pickle.load(f)
smiles = {entry['smiles']: 'train' for entry in data.values()}
choices = ['train', 'val', 'test']
weights = [0.8, 0.1, 0.1]
smiles = {entry: random.choices(choices, weights=weights)[0] for entry in smiles}
for idx in data:
    data[idx]['split'] = smiles[data[idx]['smiles']]
with open('/workspace/CombinedDataset/index.pkl', 'wb') as f:
    pickle.dump(data, f)