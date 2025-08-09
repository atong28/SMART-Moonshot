import pickle
import random

seed = 0

index = pickle.load(open('/data/nas-gpu/wang/atong/MoonshotDataset/index.pkl', 'rb'))

for idx in index:
    if index[idx]['split'] == 'test' and random.random() > 0.5:
        index[idx]['split'] = 'test_final'

pickle.dump(index, open('/data/nas-gpu/wang/atong/MoonshotDataset/index.pkl', 'wb'))
    