import logging
import pytorch_lightning as pl

def is_main_process():
    return int(os.environ.get("RANK", 0)) == 0

def init_logger(path):
    logger = logging.getLogger("lightning")
    logger.setLevel(logging.INFO if is_main_process() else logging.WARNING)
    if not logger.handlers:
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, "logs.txt")
        with open(file_path, "w"):
            pass
        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh = logging.FileHandler(file_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger

import os
import sys
import json
import shutil
from datetime import datetime
import wandb

import numpy as np
import random
import torch

from src.marina.args import parse_args
from src.marina.settings import SPECTREArgs
from src.marina.fp_loader import EntropyFPLoader
from src.marina.model import SPECTRE
from src.marina.train import train
from src.marina.test import test
from src.marina.const import DATASET_ROOT, CODE_ROOT
from src.dataset.spectre import SPECTREDataModule

def seed_everything(seed):
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    args: SPECTREArgs = parse_args()
    seed_everything(args.seed)

    # build a common results path
    today = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_path = os.path.join(
        DATASET_ROOT, "results", args.experiment_name, today
    )
    final_path = os.path.join(
        CODE_ROOT, "results", args.experiment_name, today
    )
    experiment_id = f"{args.experiment_name}_{today}"

    # rank 0: create outputs, logging, wandb
    if is_main_process() and args.train and not args.visualize:
        os.makedirs(results_path, exist_ok=True)
        logger = init_logger(results_path)
        logger.info("[Main] Parsed args:\n%s", args)

        # dump params.json
        with open(os.path.join(results_path, "params.json"), "w") as fp:
            json.dump(vars(args), fp, indent=2)

        # login to wandb
        with open("wandb_api_key.json") as kf:
            key = json.load(kf)["key"]
        wandb.login(key=key)
        wandb_run = wandb.init(
            project=args.project_name,
            name=experiment_id,
            config=vars(args),
            resume="allow",
        )
    else:
        wandb_run = None

    fp_loader = EntropyFPLoader()
    fp_loader.setup(args.out_dim, 6)
    data_module = SPECTREDataModule(args, fp_loader)
    model = SPECTRE(args, fp_loader)
    
    if args.train:
        train(args, data_module, model, results_path, wandb_run=wandb_run)
    elif args.test:
        test(args, data_module, model, results_path, ckpt_path=args.load_from_checkpoint, wandb_run=wandb_run, sweep=True)
    else:
        raise ValueError("[Main] Both --no_train and --no_test set; nothing to do!")

    if is_main_process() and args.train:
        logger.info("[Main] Moving results to final destination")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        shutil.move(results_path, final_path)
        wandb.finish()

if __name__ == "__main__":
    main()