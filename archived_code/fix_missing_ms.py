import torch
import os
from tqdm import tqdm
import pickle

DATA_ROOT = '/workspace/CombinedDataset'
index = pickle.load(open(f'{DATA_ROOT}/index.pkl', 'rb'))
for idx in tqdm(index):
    if os.path.exists(f'{DATA_ROOT}/MassSpec/{idx}.pt'):
        ms = torch.load(f'{DATA_ROOT}/MassSpec/{idx}.pt', weights_only=True)
        if ms.shape[0] == 0:
            os.remove(f'{DATA_ROOT}/MassSpec/{idx}.pt')
        index[idx]['has_mass_spec'] = False
pickle.dump(index, open(f'{DATA_ROOT}/index.pkl', 'wb'))