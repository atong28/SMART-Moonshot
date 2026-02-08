import selfies as sf
from rdkit import Chem

def smiles_to_selfies(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    can = Chem.MolToSmiles(mol)          # canonicalize
    return sf.encoder(can)
