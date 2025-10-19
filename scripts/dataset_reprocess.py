import json
import os
from tqdm import tqdm
from rdkit import Chem
import csv
import pickle
csv.field_size_limit(10485760)

LOTUS_PATH = '/data/nas-gpu/wang/atong/LotusDataset/NPOC2021/NPOC2021'
COCONUT_PATH = '/data/nas-gpu/wang/atong/CoconutDataset'
NPMRD_PATH = '/data/nas-gpu/wang/atong/NPMRDDataset'
SAVE_PATH = '/data/nas-gpu/wang/atong/MoonshotDatasetv3'

OVERRIDE = True

def canonicalize_smiles(smiles: str, keep_stereo: bool = True):
    if smiles is None or smiles == '':
        return None
    if '.' in smiles:
        smiles = max(smiles.split('.'), key=len)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=keep_stereo, canonical=True)

os.makedirs(SAVE_PATH, exist_ok=True)

def process_lotus(index: dict[str, int], all_metadata: dict[int, dict], matching, processed):
    with open(os.path.join(LOTUS_PATH, 'lotusUniqueNaturalProduct.jsonl'), 'r') as f:
        for line in tqdm(f.readlines(), desc='Processing LOTUS entries'):
            entry = json.loads(line)
            canonical_2d_smiles = canonicalize_smiles(entry['smiles'], keep_stereo=False)
            canonical_3d_smiles = canonicalize_smiles(entry['smiles'], keep_stereo=True)
            if canonical_2d_smiles in index and not OVERRIDE:
                matching += 1
                continue
            elif canonical_2d_smiles in index and OVERRIDE:
                access_idx = index[canonical_2d_smiles]
                matching += 1
            else:
                index[canonical_2d_smiles] = len(all_metadata)
                access_idx = len(all_metadata)
            processed += 1
            all_metadata[access_idx] = {
                'smiles': entry['smiles'],
                'canonical_3d_smiles': canonical_3d_smiles,
                'canonical_2d_smiles': canonical_2d_smiles,
                'lotus': {
                    'lotus_id': entry['lotus_id'],
                    'name': entry['traditional_name'],
                    'synonyms': entry['synonyms'],
                    'inchi': entry['inchi'],
                    'inchikey': entry['inchikey'],
                },
                'coconut': all_metadata.get(access_idx, {}).get('coconut', None),
                'npmrd': all_metadata.get(access_idx, {}).get('npmrd', None),
            }
    return matching, processed
            

def process_coconut(index: dict[str, int], all_metadata: dict[int, dict], matching, processed):
    reader = csv.DictReader(open(os.path.join(COCONUT_PATH, 'coconut_csv-08-2025.csv'), 'r'))
    for entry in tqdm(reader, desc='Processing COCONUT entries'):
        canonical_2d_smiles = canonicalize_smiles(entry['canonical_smiles'], keep_stereo=False)
        canonical_3d_smiles = canonicalize_smiles(entry['canonical_smiles'], keep_stereo=True)
        if canonical_2d_smiles in index and not OVERRIDE:
            matching += 1
            continue
        elif canonical_2d_smiles in index and OVERRIDE:
            access_idx = index[canonical_2d_smiles]
            matching += 1
        else:
            index[canonical_2d_smiles] = len(all_metadata)
            access_idx = len(all_metadata)
        processed += 1
        all_metadata[access_idx] = {
            'smiles': entry['canonical_smiles'],
            'canonical_3d_smiles': canonical_3d_smiles,
            'canonical_2d_smiles': canonical_2d_smiles,
            'lotus': all_metadata.get(access_idx, {}).get('lotus', None),
            'coconut': {
                'coconut_id': entry['identifier'],
                'name': entry['name'] if entry['name'] != '' else entry['iupac_name'],
                'synonyms': entry['synonyms'],
                'inchi': entry['standard_inchi'],
                'inchikey': entry['standard_inchi_key'],
            },
            'npmrd': all_metadata.get(access_idx, {}).get('npmrd', None),
        }
    return matching, processed

def process_npmrd(index: dict[str, int], all_metadata: dict[int, dict], matching, processed):
    for idx_start, idx_end in zip(range(1, 350000, 50000), range(50000, 350001, 50000)):
        json_path = os.path.join(NPMRD_PATH, f'npmrd_natural_products_NP{str(idx_start).rjust(7, "0")}_NP{str(idx_end).rjust(7, "0")}.json')
        data = json.load(open(json_path, 'r'))
        for entry in tqdm(data['np_mrd']['natural_product'], desc=f'Processing NPMRD entries {idx_start}-{idx_end}'):
            canonical_2d_smiles = canonicalize_smiles(entry['smiles'], keep_stereo=False)
            canonical_3d_smiles = canonicalize_smiles(entry['smiles'], keep_stereo=True)
            if canonical_2d_smiles is None or canonical_3d_smiles is None:
                print(f"Warning: could not parse SMILES {entry['smiles']} for NPMRD entry {entry['accession']}")
                continue
            if canonical_2d_smiles in index and not OVERRIDE:
                matching += 1
                continue
            elif canonical_2d_smiles in index and OVERRIDE:
                matching += 1
                access_idx = index[canonical_2d_smiles]
            else:
                index[canonical_2d_smiles] = len(all_metadata)
                access_idx = len(all_metadata)
            processed += 1
            all_metadata[access_idx] = {
                'smiles': entry['smiles'],
                'canonical_3d_smiles': canonical_3d_smiles,
                'canonical_2d_smiles': canonical_2d_smiles,
                'lotus': all_metadata.get(access_idx, {}).get('lotus', None),
                'coconut': all_metadata.get(access_idx, {}).get('coconut', None),
                'npmrd': {
                    'npmrd_id': entry['accession'],
                    'name': entry['name'],
                    'synonyms': entry['synonyms'],
                    'inchi': entry['inchi'],
                    'inchikey': entry['inchikey'],
                },
            }
    return matching, processed

def main():
    index = {}
    all_metadata = {}
    matching, processed = 0, 0
    matching, processed = process_lotus(index, all_metadata, matching, processed)
    matching, processed = process_coconut(index, all_metadata, matching, processed)
    matching, processed = process_npmrd(index, all_metadata, matching, processed)
    index = {v: {'smiles': k} for k, v in index.items()}
    training_index = pickle.load(open(os.path.join(SAVE_PATH, 'index.pkl'), 'rb'))
    diff = {e['smiles'] for e in training_index.values()} - {e['smiles'] for e in index.values()}
    training_index = {k: {'smiles': v} for k, v in zip(range(len(index), len(index) + len(diff)), diff)}
    index.update(training_index)
    with open(os.path.join(SAVE_PATH, 'metadata.json'), 'w') as f:
        json.dump(all_metadata, f, indent=2)
    with open(os.path.join(SAVE_PATH, 'retrieval.pkl'), 'wb') as f:
        pickle.dump(index, f)
    print(f'Processed {processed} entries, with {matching} overlapping entries.')

if __name__ == '__main__':
    main()
    