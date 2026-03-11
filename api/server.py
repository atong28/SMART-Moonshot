from typing import Optional

import os
import json
import pickle
import torch
from tqdm import tqdm

from src.modules.core.const import DATASET_ROOT, BENCHMARK_ROOT
from src.modules.marina import MARINAArgs
from src.modules.marina import MARINA
from src.modules.data.fp_loader import EntropyFPLoader
from src.modules.data.fp_utils import canonicalize_smiles
from src.modules.marina.dataset import format_inference_data
from src.modules.benchmark import filter_data

SERVER_CONFIG = {
    "model_checkpoint_dir": "/data/nas-gpu/wang/atong/SMART-Moonshot/checkpoints/website",
}

class ServerAPI:
    def __init__(self) -> None:
        params = json.load(open(os.path.join(SERVER_CONFIG["model_checkpoint_dir"], "params.json"), "r"))
        params["load_from_checkpoint"] = os.path.join(SERVER_CONFIG["model_checkpoint_dir"], "best.ckpt")
        self.args = MARINAArgs(**params)

    def initialize(self) -> None:
        self.fp_loader = EntropyFPLoader(retrieval_path=os.path.join(DATASET_ROOT, 'retrieval.pkl'))
        self.fp_loader.setup(self.args.out_dim, 6)
        self.model = MARINA(self.args, self.fp_loader)
        self.model.load_state_dict(torch.load(os.path.join(SERVER_CONFIG["model_checkpoint_dir"], "best.ckpt"))["state_dict"])
        self.model.setup_ranker()
        self.model.eval()
        self.retrieval = pickle.load(open(os.path.join(DATASET_ROOT, 'retrieval.pkl'), "rb"))
        self.metadata = json.load(open(os.path.join(DATASET_ROOT, 'metadata.json'), "r"))
        self.benchmark_data = pickle.load(open(os.path.join(BENCHMARK_ROOT, 'benchmark.pkl'), "rb"))
        self.benchmark_data = {data['npid']: data for data in self.benchmark_data.values()}

    def get_fp_for_smiles(self, smiles: str) -> torch.Tensor:
        return self.fp_loader.build_mfp_for_smiles(smiles)
    
    def rank_fp(self, fp: torch.Tensor, n: int = 10) -> list[str]:
        idxs = self.model.ranker.retrieve_idx(fp, n).detach().tolist()
        return idxs
    
    def rank_smiles(self, smiles: str, n: int = 10) -> list[str]:
        fp = self.get_fp_for_smiles(smiles)
        return [self.retrieval[idx]['smiles'] for idx in self.rank_fp(fp, n)]
    
    def cosine_sim(self, fp1: torch.Tensor, fp2: torch.Tensor) -> float:
        return torch.nn.functional.cosine_similarity(fp1, fp2, dim=0).item()
    
    def cosine_sim_smiles(self, smiles1: str, smiles2: str) -> float:
        fp1 = self.get_fp_for_smiles(smiles1)
        fp2 = self.get_fp_for_smiles(smiles2)
        return self.cosine_sim(fp1, fp2)
    
    def tanimoto_sim(self, fp1, fp2):
        fp1_bin = (fp1 > 0).int()
        fp2_bin = (fp2 > 0).int()
        intersection = (fp1_bin & fp2_bin).sum()
        union = fp1_bin.sum() + fp2_bin.sum() - intersection
        return intersection.float() / union.float()
    
    def tanimoto_sim_smiles(self, smiles1: str, smiles2: str) -> float:
        fp1 = self.get_fp_for_smiles(smiles1)
        fp2 = self.get_fp_for_smiles(smiles2)
        return self.tanimoto_sim(fp1, fp2)
    
    def retrieve_fp(self, fp: torch.Tensor, gt_fp: Optional[torch.Tensor] = None, n: int = 10) -> list[str]:
        idxs = self.rank_fp(fp, n)
        retrieval_data = {'retrievals': []}
        for idx in idxs:
            retrieval_fp = self.model.ranker.data[idx].to_dense().float()
            data = {
                'cosine_sim': self.cosine_sim(fp, retrieval_fp),
                'tanimoto_sim': self.tanimoto_sim(fp, retrieval_fp),
                'smiles': self.retrieval[idx]['smiles'],
                'retrieval_idx': idx
            }
            if gt_fp is not None:
                data['is_hit'] = self.cosine_sim(gt_fp, retrieval_fp) > 0.99
            retrieval_data['retrievals'].append(data)
        if gt_fp is not None:
            for i, retrieval in enumerate(retrieval_data['retrievals']):
                if retrieval['is_hit']:
                    retrieval_data['rank'] = i + 1
                    break
            else:
                retrieval_data['rank'] = -1
            retrieval_data['cosine_sim'] = self.cosine_sim(fp, gt_fp)
            retrieval_data['tanimoto_sim'] = self.tanimoto_sim(fp, gt_fp)
        return retrieval_data
    
    def predict(self, data: dict, gt_fp: Optional[torch.Tensor] = None, n: int = 10, restrictions: Optional[list[str]] = None) -> dict:
        if restrictions is None:
            restrictions = list(data.keys())
        inputs = format_inference_data(filter_data(data, restrictions))
        with torch.no_grad():
            output = self.model(**inputs)
        pred = torch.sigmoid(output[0])
        return self.retrieve_fp(pred, gt_fp=gt_fp, n=n)
    
    def smiles_search(self, smiles: str, n: int = 10) -> list[str]:
        fp = self.get_fp_for_smiles(smiles)
        return self.retrieve_fp(fp, n=n)

    def benchmark(self, npid: str, restrictions: Optional[dict] = None) -> dict:
        data = self.benchmark_data[npid]
        gt_fp = self.get_fp_for_smiles(data['smiles'])
        return self.predict(data['input'], gt_fp=gt_fp, n=10, restrictions=restrictions)

    def benchmark_all(self, restrictions: Optional[dict] = None) -> tuple[list[str], dict]:
        results = [self.benchmark(npid, restrictions=restrictions) for npid in tqdm(self.benchmark_data.keys())]
        stats = {
            'top_1_accuracy': sum(result['rank'] == 1 for result in results) / len(results),
            'top_5_accuracy': sum(1 <= result['rank'] <= 5 for result in results) / len(results),
            'top_10_accuracy': sum(1 <= result['rank'] <= 10 for result in results) / len(results),
            'mean_cosine_sim': sum(result['cosine_sim'] for result in results) / len(results)
        }
        return results, stats

    def format_input(self, raw_inputs: dict) -> dict:
        inputs = {}
        for input_type, input_data in raw_inputs.items():
            if input_type == 'hsqc' or input_type == 'mass_spec':
                inputs[input_type] = torch.tensor([[float(i) for i in line.split('\t')] for line in input_data.split('\n')])
                inputs[input_type] = inputs[input_type][:, [1, 0, 2]]
            elif input_type == 'h_nmr' or input_type == 'c_nmr':
                inputs[input_type] = torch.tensor([[float(i) for i in line.split('\n')] for line in input_data.split('\n')]).reshape(-1, 1)
            elif input_type == 'mw':
                inputs[input_type] = float(input_data)
        return inputs

    def custom_spectral_input(self, raw_inputs, gt_npid: str = None) -> dict:
        inputs = self.format_input(raw_inputs)
        if gt_npid is not None:
            gt_fp = self.get_fp_for_smiles(self.benchmark_data[gt_npid]['smiles'])
        else:
            gt_fp = None
        return self.predict(inputs, gt_fp=gt_fp, n=10)