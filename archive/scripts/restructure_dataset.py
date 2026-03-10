import os
import pickle
import torch
from tqdm import tqdm
import random
import itertools
import zipfile
import shutil
import traceback

random.seed(0)

def zip_directory_contents(directory: str, output_zip: str):
    with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(directory):
            for file in files:
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, directory)  # relative to directory contents
                zipf.write(full_path, arcname=relative_path)

ORIGIN_PATH = '/root/gurusmart/entropy_of_hashes_DB_updated.zip'
CHECKPOINT_PATH = '/root/gurusmart/RefactorCheckpoint.zip'
OLD_DATA_ROOT = '/workspace/OldDataset'
NEW_DATA_ROOT = '/workspace/CombinedDataset'

TRAIN_SPLIT_FRAC = 0.8
VAL_SPLIT_FRAC = 0.1

FOLDERS = ['IsoDist', 'MassSpec', 'C_NMR', 'H_NMR', 'HSQC_NMR', 'Fragments', 'HYUN_FP']

for folder in FOLDERS:
    os.makedirs(os.path.join(NEW_DATA_ROOT, folder), exist_ok=True)

path_to_unzip = CHECKPOINT_PATH if os.path.exists(CHECKPOINT_PATH) else ORIGIN_PATH

print(f'Extracting file: {path_to_unzip}')
with zipfile.ZipFile(path_to_unzip, 'r') as zip_ref:
    zip_ref.extractall(OLD_DATA_ROOT)

all_data = []
molecule_idx = 0
split_idx = 0
split_subset_idx = 0

# check if resume
if os.path.exists(os.path.join(NEW_DATA_ROOT, 'resume.pkl')):
    print('Restarting processing from checkpoint!')
    resume_data = pickle.load(open(os.path.join(NEW_DATA_ROOT, 'resume.pkl'), 'rb'))
    all_data = resume_data['all_data']
    molecule_idx = resume_data['molecule_idx']
    split_idx = resume_data['split_idx']
    split_subset_idx = resume_data['split_subset_idx']

ALL_SPLITS = list(itertools.product(('SMILES_dataset', 'OneD_Only_Dataset'), ('train', 'val', 'test')))

def get_files(datapath, folder):
    return set(os.listdir(os.path.join(datapath, folder))) if os.path.exists(os.path.join(datapath, folder)) else None

try:
    while split_idx < len(ALL_SPLITS):
        split_path = os.path.join(*ALL_SPLITS[split_idx])
        datapath = os.path.join(OLD_DATA_ROOT, split_path)
        smiles_data = pickle.load(open(os.path.join(datapath, 'SMILES', 'index.pkl'), 'rb'))
        mw_data = pickle.load(open(os.path.join(datapath, 'MW', 'index.pkl'), 'rb'))
        name_data = pickle.load(open(os.path.join(datapath, 'Chemical', 'index.pkl'), 'rb'))
        
        hsqc_files = get_files(datapath, 'HSQC')
        c_h_files = get_files(datapath, 'oneD_NMR')
        ms_files = get_files(datapath, 'MassSpec')
        id_files = get_files(datapath, 'IsotopicDistribution')
        fp_files = get_files(datapath, 'HYUN_FP')
        frag_files = get_files(datapath, 'fragments_of_different_radii')
        
        for subset_idx, idx in tqdm(
            enumerate(
                itertools.islice(smiles_data, split_subset_idx, None),
                start=split_subset_idx
            ),
            desc=f'Processing {split_path}',
            initial=split_subset_idx,
            total=len(smiles_data)
        ):
            filename = f'{idx}.pt'
            molecule_data = {
                'has_hsqc': False,
                'has_c_nmr': False,
                'has_h_nmr': False,
                'has_mass_spec': False,
                'has_iso_dist': False,
                'mw': None,
                'smiles': None,
                'hyun_fp': None,
                'name': None,
                'split': ALL_SPLITS[split_idx][1]
            }
            if hsqc_files is not None and filename in hsqc_files:
                molecule_data['has_hsqc'] = True
                torch.save(
                    torch.load(os.path.join(datapath, 'HSQC', filename), weights_only=True),
                    os.path.join(NEW_DATA_ROOT, 'HSQC_NMR', f'{molecule_idx}.pt')
                )
            if c_h_files is not None and filename in c_h_files:
                molecule_data['has_c_nmr'] = True
                molecule_data['has_h_nmr'] = True
                c, h = torch.load(os.path.join(datapath, 'oneD_NMR', filename), weights_only=True)
                torch.save(
                    c,
                    os.path.join(NEW_DATA_ROOT, 'C_NMR', f'{molecule_idx}.pt')
                )
                torch.save(
                    h,
                    os.path.join(NEW_DATA_ROOT, 'H_NMR', f'{molecule_idx}.pt')
                )
            if ms_files is not None and filename in ms_files:
                molecule_data['has_mass_spec'] = True
                torch.save(
                    torch.load(os.path.join(datapath,'MassSpec', filename), weights_only=True),
                    os.path.join(NEW_DATA_ROOT, 'MassSpec', f'{molecule_idx}.pt')
                )
            if id_files is not None and filename in id_files:
                molecule_data['has_iso_dist'] = True
                torch.save(
                    torch.load(os.path.join(datapath,'IsotopicDistribution', filename), weights_only=True),
                    os.path.join(NEW_DATA_ROOT, 'IsoDist', f'{molecule_idx}.pt')
                )
            if fp_files is not None and filename in fp_files:
                torch.save(
                    torch.load(os.path.join(datapath, 'HYUN_FP', filename), weights_only=True),
                    os.path.join(NEW_DATA_ROOT, 'HYUN_FP', f'{molecule_idx}.pt')
                )
            if frag_files is not None and filename in frag_files:
                torch.save(
                    torch.load(os.path.join(datapath, 'fragments_of_different_radii', filename), weights_only=True),
                    os.path.join(NEW_DATA_ROOT, 'Fragments', f'{molecule_idx}.pt')
                )
            if idx in mw_data:
                molecule_data['has_mw'] = True
                molecule_data['mw'] = mw_data[idx]
            molecule_data['smiles'] = smiles_data[idx]
            if idx not in name_data:
                print(f'Name not found for idx {idx}')
            else:
                molecule_data['name'] = name_data[idx]
            #split_float = random.random()
            #if split_float > TRAIN_SPLIT_FRAC:
            #    if split_float > TRAIN_SPLIT_FRAC + VAL_SPLIT_FRAC:
            #        molecule_data['split'] = 'test'
            #    else:
            #        molecule_data['split'] = 'val'
            all_data.append(molecule_data)
            molecule_idx += 1
            
        split_idx += 1

    pickle.dump(dict(enumerate(all_data)), open(os.path.join(NEW_DATA_ROOT, 'index.pkl'), 'wb'))
    if os.path.exists(os.path.join(NEW_DATA_ROOT, 'resume.pkl')):
        os.remove(os.path.join(NEW_DATA_ROOT, 'resume.pkl'))
    shutil.copy2(os.path.join(OLD_DATA_ROOT, 'count_hashes_under_radius_10.pkl'), os.path.join(NEW_DATA_ROOT, 'count_hashes_under_radius_10.pkl'))
    zip_directory_contents(NEW_DATA_ROOT, '/root/gurusmart/MoonshotDataset_v0.1.zip')
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
    print('Completed!')
    
except (KeyboardInterrupt, Exception):
    print(traceback.format_exc())
    print('Error occurred or Ctrl+C pressed. Cleaning up...')
    resume_data = {
        'all_data': all_data,
        'molecule_idx': molecule_idx,
        'split_idx': split_idx,
        'split_subset_idx': subset_idx
    }
    pickle.dump(resume_data, open(os.path.join(NEW_DATA_ROOT, 'resume.pkl'), 'wb'))
    zip_directory_contents(OLD_DATA_ROOT, '/root/gurusmart/RefactorCheckpoint.zip')