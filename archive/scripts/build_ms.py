import os
import json
from tqdm import tqdm
import pickle
from collections import defaultdict
import torch
os.makedirs('/workspace/CombinedDataset/MassSpec', exist_ok=True)

MS_DATA_PATH = '/root/gurusmart/json'
all_data = []
for filepath in tqdm(os.listdir(MS_DATA_PATH)):
    all_data += json.load(open(os.path.join(MS_DATA_PATH, filepath), 'r'))

data = pickle.load(open('/workspace/CombinedDataset/index.pkl', 'rb'))

inverse_map = defaultdict(list)
for k, v in data.items():
    inverse_map[v['smiles']].append(k)

for entry in tqdm(all_data):
    idxs = inverse_map.get(entry['SMILES'])
    if idxs is None:
        print(f'Failed to match {entry["SMILES"]}')
        continue
    for idx in idxs:
        with open(os.path.join('/workspace/CombinedDataset', 'MassSpec', f'{idx}.pt'), 'wb') as f:
            torch.save(torch.tensor(entry['peaks'], dtype=torch.float64), f)
    