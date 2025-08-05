import pickle
from rdkit import Chem, RDLogger
from tqdm import tqdm
RDLogger.DisableLog('rdApp.*')
index = pickle.load(open('/data/nas-gpu/wang/atong/MoonshotDataset/index.pkl', 'rb'))

for idx in tqdm(index):
    smiles = index[idx]['smiles']
    mol = Chem.MolFromSmiles(smiles)
    smi = Chem.MolToSmiles(mol, isomericSmiles=False) # remove stereochemistry information
    mol = Chem.MolFromSmiles(smi)
    index[idx]['inchi'] = Chem.MolToInchi(mol)
    index[idx]['inchi_key'] = Chem.inchi.InchiToInchiKey(index[idx]['inchi'])

pickle.dump(index, open('/data/nas-gpu/wang/atong/MoonshotDataset/index.pkl', 'wb'))