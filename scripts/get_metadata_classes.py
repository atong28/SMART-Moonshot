import requests
import json
import pickle
from tqdm import tqdm
import time
import urllib.parse

def get_np_classes(smiles):
    try:
        # smiles = 'CC1C(O)CC2C1C(OC1OC(COC(C)=O)C(O)C(O)C1O)OC=C2C(O)=O'
        url = f"https://npclassifier.gnps2.org/classify?smiles={urllib.parse.quote(smiles, safe='')}"
        response = requests.get(url)
        json_dat = json.loads(response.content)
        class_results = json_dat['class_results']
        superclass_results = json_dat['superclass_results']
        pathway_results = json_dat['pathway_results']
        return pathway_results, superclass_results, class_results
    except Exception as e:
        print(smiles)
        print(response.content)
        return None, None, None

metadata = pickle.load(open('/data/nas-gpu/wang/atong/MoonshotDataset/rankingset_metadata.pkl', 'rb'))
index = {}

for idx, entry in enumerate(tqdm(metadata)):
    smiles = entry[0]
    index[idx] = {'smiles': smiles}
    if '.' in smiles:
        smi_list = smiles.split('.')
        smiles = max(smi_list, key=len)
    pathway_results, superclass_results, class_results = get_np_classes(smiles)
    index[idx]['np_pathway'] = pathway_results
    index[idx]['np_superclass'] = superclass_results
    index[idx]['np_class'] = class_results

pickle.dump(index, open('/data/nas-gpu/wang/atong/MoonshotDataset/rankingset_meta.pkl', 'wb'))