from typing import Any, List
import numpy as np
import torch
import os
import pickle
import json
from collections import OrderedDict
from tqdm import tqdm
from rdkit import Chem

from wandb.sdk.wandb_run import Run
import wandb

from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs import ConvertToNumpyArray
from .marina import MARINAArgs,MARINADataModule, MARINA
from .spectre import SPECTREArgs, SPECTREDataModule, SPECTRE
from .log import get_logger
from .core.const import BENCHMARK_ROOT, DATASET_ROOT, INPUT_TYPES
from .data.fp_loader import EntropyFPLoader


_gen = GetMorganGenerator(radius=2, fpSize=2048)

def cos_sim(pred, target):
    return torch.dot(pred, target) / (torch.norm(pred) * torch.norm(target))

def tanimoto_sim(pred, target):
    pred_bin = (pred > 0).int()
    target_bin = (target > 0).int()
    intersection = (pred_bin & target_bin).sum()
    union = pred_bin.sum() + target_bin.sum() - intersection
    return intersection.float() / union.float()

def get_mfp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    fp = _gen.GetFingerprint(mol)
    arr = np.zeros((2048,), dtype=np.float32)
    ConvertToNumpyArray(fp, arr)
    return torch.tensor(arr, dtype=torch.float32)

def load_model(args: MARINAArgs | SPECTREArgs, model: MARINA | SPECTRE) -> None:
    if args.project_name == 'MARINA':
        model.load_state_dict(torch.load(args.load_from_checkpoint)['state_dict'])
    elif args.project_name == 'SPECTRE':
        state_dict = torch.load(args.load_from_checkpoint)['state_dict']
        encoders = []
        for k, v in model.state_dict().items():
            if 'sin_term' in k or 'cos_term' in k:
                encoders.append((k, v))
        state = [(k, v) for k, v in state_dict.items()]
        state = [state[0]] + encoders + state[1:]
        state_dict = OrderedDict(state)
        model.load_state_dict(state_dict)
    model.setup_ranker()
    model.eval()
    
def filter_data(data: dict[int, Any], restrictions: List[INPUT_TYPES]) -> dict[int, Any]:
    return {k: v for k, v in data.items() if k in restrictions}

def benchmark(
    args: MARINAArgs | SPECTREArgs,
    data_module: MARINADataModule | SPECTREDataModule,
    model: MARINA | SPECTRE,
    fp_loader: EntropyFPLoader,
    wandb_run: Run | None = None,
    load_from_checkpoint: str | None = None,
) -> None:
    """
    Benchmark a MARINA or SPECTRE model.
    """
    restrictions = args.input_types if args.restrictions is None else args.restrictions
    if load_from_checkpoint is not None:
        load_model(args, model)
    if BENCHMARK_ROOT is None:
        raise ValueError('Benchmarking is not supported on this setup')
    logger = get_logger(__file__)
    logger.info(f'[Benchmark] Benchmarking model {model.__class__.__name__}')
    metadata = json.load(open(os.path.join(DATASET_ROOT, "metadata.json"), 'r'))
    meta_smi_to_idx = {entry['canonical_2d_smiles']: int(idx) for idx, entry in metadata.items()}
    benchmark_data: dict[int, Any] = pickle.load(open(os.path.join(BENCHMARK_ROOT, "benchmark.pkl"), 'rb'))
    benchmark_data = {k: v for k, v in benchmark_data.items() if v['split'] == args.benchmark_split}
    for idx, entry in tqdm(benchmark_data.items()):
        inputs = data_module.format_inference_data(filter_data(entry['input'], restrictions))
        output = model(**inputs)
        pred = torch.sigmoid(output[0])
        idxs = model.ranker.retrieve_idx(pred, 10).tolist()
        sfp = fp_loader.build_mfp_for_smiles(entry['smiles'])
        sfp = sfp / torch.norm(sfp)
        mfp = get_mfp(entry['smiles'])
        gt_retrieval_idx = meta_smi_to_idx.get(entry['smiles'], None)
        retrievals = {}
        near_identical_matches = []
        for idx_k in range(10):
            retrieval_idx = idxs[idx_k]
            retrieval_sfp = model.ranker.data[retrieval_idx].to_dense().float()
            retrieval_mfp = get_mfp(metadata[str(retrieval_idx)]['canonical_2d_smiles'])
            cos_sim_sfp = cos_sim(sfp, retrieval_sfp)
            near_identical_matches.append(cos_sim_sfp > 0.99)
            retrievals[idx_k] = {
                'retrieval_idx': retrieval_idx,
                'retrieval_sfp': retrieval_sfp,
                'retrieval_mfp': get_mfp(metadata[str(retrieval_idx)]['canonical_2d_smiles']),
                'cosine_sim_sfp': cos_sim_sfp,
                'tani_sim_sfp': tanimoto_sim(sfp, retrieval_sfp),
                'cosine_sim_mfp': cos_sim(mfp, retrieval_mfp),
                'tani_sim_mfp': tanimoto_sim(mfp, retrieval_mfp)
            }
        benchmark_data[idx]['predictions'] = {
            'pred_sfp': pred / torch.norm(pred),
            'sfp': sfp,
            'mfp': mfp,
            'cosine_sim': cos_sim(pred, sfp),
            'retrievals': retrievals,
            'retrieval_idx': gt_retrieval_idx,
            'dereplication_topk': {k: any(near_identical_matches[:k]) for k in range(1, 11)}
        }
    pickle.dump(benchmark_data, open(os.path.join(BENCHMARK_ROOT, 'benchmarks', f"{args.experiment_name}_benchmark_results.pkl"), 'wb'))
    logger.info(f'[Benchmark] Benchmarking completed')
    logger.info(f'[Benchmark] Average cosine similarity: {torch.mean(torch.tensor([entry["predictions"]["cosine_sim"] for entry in benchmark_data.values()])).item()}')
    dereplication_topk = {k: [entry["predictions"]["dereplication_topk"][k] for entry in benchmark_data.values() if entry["predictions"]["dereplication_topk"][k] is not None] for k in range(1, 11)}
    logger.info(f'[Benchmark] Average dereplication top1: {sum(dereplication_topk[1])} out of {len(dereplication_topk[1])} available ({sum(dereplication_topk[1]) / len(dereplication_topk[1]) * 100:.2f}%)')
    logger.info(f'[Benchmark] Average dereplication top5: {sum(dereplication_topk[5])} out of {len(dereplication_topk[5])} available ({sum(dereplication_topk[5]) / len(dereplication_topk[5]) * 100:.2f}%)')
    logger.info(f'[Benchmark] Average dereplication top10: {sum(dereplication_topk[10])} out of {len(dereplication_topk[10])} available ({sum(dereplication_topk[10]) / len(dereplication_topk[10]) * 100:.2f}%)')

    if wandb_run is not None:
        mean_cos = torch.mean(torch.tensor([entry["predictions"]["cosine_sim"] for entry in benchmark_data.values()])).item()
        n1, n5, n10 = len(dereplication_topk[1]), len(dereplication_topk[5]), len(dereplication_topk[10])
        wandb.log({
            "benchmark/mean_cos": mean_cos,
            "benchmark/dereplication_top1_pct": sum(dereplication_topk[1]) / n1 * 100 if n1 else 0.0,
            "benchmark/dereplication_top5_pct": sum(dereplication_topk[5]) / n5 * 100 if n5 else 0.0,
            "benchmark/dereplication_top10_pct": sum(dereplication_topk[10]) / n10 * 100 if n10 else 0.0,
            "benchmark/dereplication_top1_count": sum(dereplication_topk[1]),
            "benchmark/dereplication_top5_count": sum(dereplication_topk[5]),
            "benchmark/dereplication_top10_count": sum(dereplication_topk[10]),
        })