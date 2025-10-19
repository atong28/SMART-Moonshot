import shutil
import pickle

index = pickle.load(open('/workspace/index.pkl', 'rb'))
smiles_dict = {index[idx]['smiles']: {
    'has_c_nmr': False,
    'has_h_nmr': False,
} for idx in index}
for idx in index:
    if index[idx]['has_c_nmr']:
        smiles_dict[index[idx]['smiles']]['has_c_nmr'] = True
        smiles_dict[index[idx]['smiles']]['idx_c_nmr'] = idx
    if index[idx]['has_h_nmr']:
        smiles_dict[index[idx]['smiles']]['has_h_nmr'] = True
        smiles_dict[index[idx]['smiles']]['idx_h_nmr'] = idx
for idx in index:
    if index[idx]['has_c_nmr']:
        continue
    if smiles_dict[index[idx]['smiles']]['has_c_nmr']:
        nmr_idx = smiles_dict[index[idx]['smiles']]['idx_c_nmr']
        shutil.copy2(f'/workspace/C_NMR/{nmr_idx}.pt', f'/workspace/C_NMR/{idx}.pt')
        index[idx]['has_c_nmr'] = True
for idx in index:
    if index[idx]['has_h_nmr']:
        continue
    if smiles_dict[index[idx]['smiles']]['has_h_nmr']:
        nmr_idx = smiles_dict[index[idx]['smiles']]['idx_h_nmr']
        shutil.copy2(f'/workspace/H_NMR/{nmr_idx}.pt', f'/workspace/H_NMR/{idx}.pt')
        index[idx]['has_h_nmr'] = True
pickle.dump(index, open('/workspace/index.pkl', 'wb'))