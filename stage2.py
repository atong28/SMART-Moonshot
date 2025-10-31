#!/usr/bin/env python3
import os
import sys
import json
import shutil
import logging
from datetime import datetime

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from src.molemcl.args import parse_args
from src.molemcl.settings import MoleMCLArgs
from src.molemcl.train import train as run_train
from src.molemcl.test import test as run_test

torch.set_float32_matmul_precision('high')

def is_main_process():
    return int(os.environ.get("RANK", 0) or 0) == 0

def init_logger(path: str):
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
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger

def seed_everything(seed: int):
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def main():
    args: MoleMCLArgs = parse_args()
    seed_everything(args.seed)

    # Timestamped results folders (like Stage 1)
    today = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_path = os.path.join(args.output_dir, args.experiment_name, today)
    final_path = os.path.join("results", args.experiment_name, today)  # mirror into repo-local results
    experiment_id = f"{args.experiment_name}_{today}"

    # rank 0: files + logging + wandb
    if is_main_process() and (args.train or args.test):
        os.makedirs(results_path, exist_ok=True)
        logger = init_logger(results_path)
        logger.info("[Stage2] Parsed args:\n%s", args)

        # dump params.json
        with open(os.path.join(results_path, "params.json"), "w") as fp:
            json.dump(vars(args), fp, indent=2)

        # WandB: reads key from env (WANDB_API_KEY) or prior login
        wandb_logger = WandbLogger(
            project=args.project_name,
            name=experiment_id,
            save_dir=results_path,
            log_model=False,
        )
    else:
        logger = logging.getLogger("lightning")
        wandb_logger = None

    # Train / Test
    if args.train:
        run_train(args, pl_logger=wandb_logger)
    if args.test:
        # if you want to test the last/best checkpoint automatically, set args.load_from_checkpoint/ckpt_path
        run_test(args, pl_logger=wandb_logger)

    # Move run folder into final destination (like Stage 1)
    if is_main_process():
        logger.info("[Stage2] Moving results to final destination")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        shutil.move(results_path, final_path)

if __name__ == "__main__":
    main()
