import pickle
# TODO: Fix the hardcoded paths
data = pickle.load(open('/data/nas-gpu/wang/atong/MoonshotDatasetv2/index.pkl', 'rb'))
for idx, entry in data.items():
    smiles = entry["smiles"]
    if '.' in smiles:
        smi_list = smiles.split('.')
        smiles = max(smi_list, key=len)
    data[idx]['smiles'] = smiles
# TODO: Fix the hardcoded paths
pickle.dump(data, open('/data/nas-gpu/wang/atong/MoonshotDatasetv2/index.pkl', 'wb'))