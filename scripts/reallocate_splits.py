import pickle
import random

random.seed(0)

with open('/workspace/index.pkl', 'rb') as f:
    data = pickle.load(f)
choices = ['train', 'val', 'test']
weights = [0.8, 0.1, 0.1]
for idx in data:
    data[idx]['split'] = random.choices(choices, weights=weights)[0]
with open('/workspace/index.pkl', 'wb') as f:
    pickle.dump(data, f)