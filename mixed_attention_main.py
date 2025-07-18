import argparse
import logging
import os
import sys
import json
import shutil
from datetime import datetime
import json
import wandb

import numpy as np
import random
import torch
import pytorch_lightning as pl

from mixed_attention.src.fp_loaders import get_fp_loader
from mixed_attention.src.settings import Args
from mixed_attention.src.dataset import MoonshotDataModule
from mixed_attention.src.model import build_model
from mixed_attention.train import train
from mixed_attention.test import test
from mixed_attention.debug import debug

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
    
    add_bool_flag(parser, 'develop', False)

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

def seed_everything(seed):
    """
    Set the random seed for reproducibility.
    """
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)

    # ---- everyone computes the same path string ----
    today = datetime.now().strftime("%Y-%m-%d")
    results_path = os.path.join(args.data_root,  "results", args.experiment_name, today)
    final_path = os.path.join(args.code_root,  "results", args.experiment_name, today)
    experiment_id = f"{args.experiment_name}_{today}"

    # ---- local rank 0 does all of the side‐effects ----
    if is_main_process():
        os.makedirs(results_path, exist_ok=True)

        # initialize file logger
        logger = init_logger(results_path)
        logger.info("[main] parsed args:\n%s", args)

        # dump params.json
        with open(os.path.join(results_path, "params.json"), "w") as fp:
            json.dump(vars(args), fp, indent=2)

        # login & start wandb
        with open("/root/gurusmart/wandb_api_key.json") as kf:
            wandb.login(key=json.load(kf)["key"])
        wandb_run = wandb.init(
            project="SPECTRE",
            name=experiment_id,
            config=vars(args),
            resume="allow",
        )
    else:
        # other ranks skip everything
        wandb_run = None

    # ---- now every rank instantiates data+model+trainer normally ----
    fp_loader = get_fp_loader(args)
    data_module = MoonshotDataModule(args, results_path, fp_loader)
    model = build_model(
        args,
        optional_inputs=(set(args.requires) != set(args.input_types)),
        fp_loader=fp_loader,
        combinations_names=data_module.combinations_names,
    )

    if args.develop:
        if is_main_process():
            logger.info("[Main] Entering develop mode")
        debug(args, data_module, model)
        sys.exit(0)

    if args.train:
        train(args, data_module, model, results_path, wandb_run=wandb_run)
    elif args.test:
        test(args, data_module, results_path, model, ckpt_path=args.load_from_checkpoint, wandb_run=wandb_run)
    else:
        raise ValueError("[Main] Train and test both disabled, nothing to do!")

    # ---- only rank0 does the final copy + wandb.finish ----
    if is_main_process():
        logger.info("[Main] Copying results to final destination")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        shutil.copytree(results_path, final_path, dirs_exist_ok=True)
        wandb.finish()