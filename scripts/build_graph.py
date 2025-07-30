from rdkit import Chem
from torch_geometric.data import Data
import torch
import pickle
from tqdm import tqdm

def atom_features(atom):
    return torch.tensor([
        atom.GetAtomicNum(),                     # Atomic number
        atom.GetTotalDegree(),                   # Total bonds
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),
        int(atom.GetIsAromatic())
    ], dtype=torch.float)

def bond_features(bond):
    bt = bond.GetBondType()
    bond_type = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3,
    }
    return torch.tensor([
        bond_type.get(bt, -1),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing())
    ], dtype=torch.float)

def smiles_to_graph(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # skip invalid

    # Nodes (atoms)
    atom_feats = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.stack(atom_feats, dim=0)  # [num_nodes, node_feat_dim]

    # Edges (bonds)
    edge_index_list = []
    edge_attr_list  = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index_list += [[i, j], [j, i]]
        edge_feat = bond_features(bond)
        edge_attr_list += [edge_feat, edge_feat]

    if edge_index_list:
        # transpose 2D tensor
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).mT.contiguous()  # [2, num_edges]
        edge_attr  = torch.stack(edge_attr_list, dim=0)                               # [num_edges, edge_feat_dim]
    else:
        # no bonds → empty graph
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, 3), dtype=torch.float)  # same bond_feat dim

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)


index = pickle.load(open('/workspace/index.pkl', 'rb'))

for idx, entry in tqdm(index.items()):
    try:
        index[idx]["graph"] = smiles_to_graph(entry["smiles"])
    except Exception as e:
        print(entry["smiles"])
        raise e

pickle.dump(index, open('/workspace/index.pkl', 'wb'))