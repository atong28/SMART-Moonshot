import torch
import pickle
from tqdm import tqdm

index = pickle.load(open('/workspace/index.pkl', 'rb'))

for idx, entry in tqdm(index.items()):
    if 'graph' in entry:
        del index[idx]["graph"]

pickle.dump(index, open('/workspace/index.pkl', 'wb'))