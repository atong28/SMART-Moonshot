from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import pickle
from tqdm import tqdm
import re

def smiles_to_formula(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f'Failed: {smiles}')
        return None
    formula = rdMolDescriptors.CalcMolFormula(mol)
    return parse_charge(formula)[0]

def parse_charge(raw_formula: str):
    match = re.match(r"^([A-Za-z0-9]+)([+-]\d*)?$", raw_formula.strip())
    if not match:
        raise ValueError(f"Invalid formula: {raw_formula}")
    
    formula = match.group(1)
    charge_str = match.group(2)

    if charge_str:
        if charge_str in ('+', '-'):
            charge = 1 if charge_str == '+' else -1
        else:
            charge = int(charge_str)
    else:
        charge = 0

    return formula, charge

with open('/workspace/index.pkl', 'rb') as f:
    data = pickle.load(f)

for idx, entry in tqdm(data.items()):
    data[idx]['formula'] = smiles_to_formula(data[idx]['smiles'])

with open('/workspace/index.pkl', 'wb') as f:
    pickle.dump(data, f)