import argparse
import logging
import os
import sys
import json
import shutil
from datetime import datetime
import json
import wandb

from cross_attention.fp_loaders import get_fp_loader
from cross_attention.settings import Args
from cross_attention.dataset import MoonshotDataModule
from cross_attention.model import build_model
from cross_attention.train import train
from cross_attention.test import test

def is_main_process():
    return int(os.environ.get("RANK", 0)) == 0

def init_logger(path):
    logger = logging.getLogger("lightning")
    if is_main_process():
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
    if not logger.handlers:
        file_path = os.path.join(path, "logs.txt")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as fp:
            pass

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(file_path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger

def add_bool_flag(parser: argparse.ArgumentParser, name: str, default: bool):
    if default:
        parser.add_argument(f'--no_{name}', dest=name, action='store_false')
    else:
        parser.add_argument(f'--{name}', dest=name, action='store_true')
    parser.set_defaults(**{name: default})

def parse_args() -> Args:
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_name')
    parser.add_argument('--code_root')
    parser.add_argument('--inference_root')
    parser.add_argument('--data_root')
    parser.add_argument('--split', choices=['train', 'val', 'test'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--load_from_checkpoint')

    parser.add_argument('--input_types', nargs='+', choices=['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mw', 'iso_dist'])
    parser.add_argument('--requires', nargs='+', choices=['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mw', 'iso_dist'])

    add_bool_flag(parser, 'debug', False)
    add_bool_flag(parser, 'persistent_workers', True)
    add_bool_flag(parser, 'validate_all', False)
    add_bool_flag(parser, 'use_cached_datasets', True)
    add_bool_flag(parser, 'use_peak_values', False)
    add_bool_flag(parser, 'save_params', True)
    add_bool_flag(parser, 'freeze_weights', False)
    add_bool_flag(parser, 'use_jaccard', False)
    add_bool_flag(parser, 'rank_by_soft_output', True)
    add_bool_flag(parser, 'rank_by_test_set', False)
    add_bool_flag(parser, 'train', True)
    add_bool_flag(parser, 'test', True)

    parser.add_argument('--fp_type', choices=['Entropy', 'HYUN', 'Normal'])
    parser.add_argument('--fp_radius', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--jittering', type=float)

    parser.add_argument('--dim_model', type=int)
    parser.add_argument('--dim_coords', type=int, nargs=3)
    parser.add_argument('--heads', type=int)
    parser.add_argument('--layers', type=int)
    parser.add_argument('--ff_dim', type=int)
    parser.add_argument('--out_dim', type=int)
    parser.add_argument('--accumulate_grad_batches_num', type=int)

    parser.add_argument('--dropout', type=float)
    parser.add_argument('--ranking_set_path')

    parser.add_argument('--lr', type=float)
    parser.add_argument('--noam_factor', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--l1_decay', type=float)
    parser.add_argument('--scheduler', choices=['attention'])
    parser.add_argument('--warm_up_steps', type=int)

    args = parser.parse_args()
    if args.load_from_checkpoint:
        checkpoint_dir = os.path.dirname(args.load_from_checkpoint)
        params_path = os.path.join(checkpoint_dir, 'params.json')
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"No params.json found in checkpoint directory: {params_path}")
        
        with open(params_path, 'r') as f:
            checkpoint_args_dict = json.load(f)

        for k, v in checkpoint_args_dict.items():
            setattr(args, k, v)
    args_dict = {k: v for k, v in vars(args).items() if v is not None}

    return Args(**args_dict)

if __name__ == "__main__":
    with open('/root/gurusmart/wandb_api_key.json', 'r') as f:
        wandb.login(key=json.load(f)['key'])
    args = parse_args()
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_path = os.path.join(args.data_root, 'results', args.experiment_name, now)
    final_results_path = os.path.join(args.code_root, 'results', args.experiment_name, now)
    experiment_name = args.experiment_name + f'_{now}'
    while os.path.exists(results_path):
        results_path += '_copy'
    os.makedirs(results_path, exist_ok=True)
    logger = init_logger(results_path)    
    logger.info('[Main] Parsed args:')
    logger.info(args)
    with open(os.path.join(results_path, 'params.json'), 'w') as f:
        json.dump(args.__dict__, f)
    fp_loader = get_fp_loader(args)
    data_module = MoonshotDataModule(args, results_path, fp_loader)
    optional_inputs = set(args.requires) != set(args.input_types)
    model = build_model(args, optional_inputs, fp_loader, combinations_names=data_module.combinations_names)
    if args.train:
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            wandb_run = wandb.init(
                project='SPECTRE',
                name=experiment_name,
                config=args.__dict__,
                resume="allow",
            )
        else:
            wandb_run = None
        train(args, data_module, model, results_path, wandb_run=wandb_run)
    elif args.test:
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            test(args, data_module, results_path, model, ckpt_path=args.load_from_checkpoint, wandb_run=None)
    else:
        raise ValueError('Both train and test are disabled, nothing to do!')
    
    logger.info('[Main] Experiment complete! Copying the results folder to its final destination')
    logger.info(f'[Main] Copying {results_path} to {final_results_path}')
    os.makedirs(os.path.dirname(final_results_path), exist_ok=True)
    shutil.copytree(results_path, final_results_path, dirs_exist_ok=True)
    logger.info('[Main] Done!')
    wandb.finish()