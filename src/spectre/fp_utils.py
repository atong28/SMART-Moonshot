
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import pickle, os, tqdm, torch
import multiprocessing, collections
import numpy as np
from collections import defaultdict  
from rdkit.Chem import rdFingerprintGenerator

from .const import DATASET_ROOT
# DATASETS = ["OneD_Only_Dataset", "SMILES_dataset"]
# DATASET_INDEX_SOURCE = ["oneD_NMR" , "HSQC"]
RADIUS_UPPER_LIMIT = 10

gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS_UPPER_LIMIT)
ao = rdFingerprintGenerator.AdditionalOutput()
ao.AllocateBitInfoMap()

smiles_dict = pickle.load(open(os.path.join(DATASET_ROOT, 'index.pkl'), 'rb'))
smiles_dict = {idx: smiles_dict[entry]['smiles'] for idx, entry in enumerate(smiles_dict)}

def compute_entropy(data, total_dataset_size):
    p = data / total_dataset_size
    entropy = p * np.log2(np.clip(p,1e-7 ,1))  +  (1-p) * np.log2(np.clip(1-p,1e-7 ,1))
    return entropy

def get_bitInfos_for_each_atom_idx(SMILES, ignoreAtoms=[]):
    """
    Args:
        SMILES (_type_): 

    Returns:
        atom_to_bit_infos (dict): 
         mapping atom index to a list of tuples; each tuple has (bit id, atom symbol, frag_smiles, radius)
    """
    
    mol = Chem.MolFromSmiles(SMILES)
    if mol is None:
        print(f"Failed to parse {SMILES}")
        # raise ValueError(f"Failed to parse {smiles}")
        return None, None
    # Chem.Kekulize(mol, clearAromaticFlags=True)

    # Compute Morgan fingerprint with radius 
    fp = gen.GetSparseFingerprint(mol, additionalOutput=ao)
    # print("active bits: ", len(torch.tensor(fp).nonzero())) 
    info = ao.GetBitInfoMap()          
    
    atom_to_bit_infos = defaultdict(list)
    all_bit_infos = set()
    for bit_id, atom_envs in info.items():
        for atom_idx, curr_radius in atom_envs:
            if atom_idx in ignoreAtoms:
                continue
            # Get the circular environment as a subgraph
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, curr_radius, atom_idx)
            submol = Chem.PathToSubmol(mol, env)
            smiles = Chem.MolToSmiles(submol, canonical=True) # this is canonical in terms of fragment, so it is related to the bond/atom index mapping
            
            bit_info = (bit_id, mol.GetAtomWithIdx(atom_idx).GetSymbol(), smiles, curr_radius)
            atom_to_bit_infos[atom_idx].append(bit_info)
            all_bit_infos.add(bit_info)
            
    return atom_to_bit_infos, all_bit_infos

# step 1: find all fragments of the entire training set
def count_circular_substructures(smiles, ignoreAtoms=[]):
    bit_info_counter = defaultdict(int) 
    atom_to_bit_infos, all_bit_infos = get_bitInfos_for_each_atom_idx(smiles, ignoreAtoms)
    if atom_to_bit_infos is None:
        return bit_info_counter
    for bit_info in all_bit_infos:
        bit_info_counter[bit_info] = 1
    return bit_info_counter

def save_frags_for_file(f):
    number_part = f.split('/')[-1]
    idx = int(number_part.split(".")[0])
    smile = smiles_dict[idx]
    bit_info_counter = count_circular_substructures(smile)
    torch.save(list(bit_info_counter.keys()), f)
    return None

def merge_counts(counts_list):
    """Merges multiple word count dictionaries."""
    total_count = collections.Counter()
    for count in counts_list:
        total_count.update(count)
        
    return total_count

def get_all_fragments_from_all_smiles_in_retrieval_dataset():
    # Load the dataset
    count_results = []
    raise NotImplementedError()
    with open(f'INFERENCE_ROOT/coconut_loutus_hyun_training/inference_metadata_latest_RDkit.pkl', 'rb') as file:
        smiles_and_names = pickle.load(file)
        all_SMILES = [x[0] for x in smiles_and_names]
        
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for count in tqdm.tqdm(pool.imap_unordered(count_circular_substructures, all_SMILES), total=len(all_SMILES), desc="Processing files"):
            count_results.append(count)
            
    final_frags_count = merge_counts(count_results)
       
    # save a dictionary of fragment to index
    save_path = os.path.join(DATASET_ROOT, f"count_hashes_under_radius_{RADIUS_UPPER_LIMIT}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(final_frags_count, f)
   
def generate_frags():
    os.makedirs(os.path.join(DATASET_ROOT, 'Fragments'), exist_ok=True)
    files = [os.path.join(DATASET_ROOT, 'Fragments', f'{idx}.pt') for idx in smiles_dict]
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(save_frags_for_file, files), total=len(files), desc=f"Processing files"):
            pass

if __name__ == "__main__":
    generate_frags()