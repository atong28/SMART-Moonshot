import logging
import warnings
import os
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings(
    "ignore",
    message="The PyTorch API of nested tensors is in prototype stage",
    category=UserWarning,
    module="torch.nn.modules.transformer"
)
warnings.filterwarnings(
    "ignore",
    message=r"It is recommended to use `self\.log\('.*', \.\.\., sync_dist=True\)` when logging on epoch level",
    category=UserWarning,
    module="pytorch_lightning.trainer.connectors.logger_connector.result"
)
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

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

import sys
import json
import shutil
from datetime import datetime
import wandb
import pytorch_lightning as pl

import numpy as np
import random
import torch
from src.marina.core.args import parse_args
from src.marina.core.settings import MARINAArgs
from src.marina.data.fp_loader import make_fp_loader
from src.marina.arch.model import SPECTRE
from src.marina.train import train
from src.marina.test import test
from src.marina.core.const import DATASET_ROOT, WANDB_API_KEY_FILE, PVC_ROOT
from src.marina.data.dataset import SPECTREDataModule

def seed_everything(seed):
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    args: MARINAArgs = parse_args()
    seed_everything(args.seed)
    today = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_path = os.path.join(DATASET_ROOT, "results", args.experiment_name, today)
    final_path = os.path.join(PVC_ROOT, "results", args.experiment_name, today)
    experiment_id = f"{args.experiment_name}_{today}"

    if is_main_process() and args.train and not args.visualize:
        os.makedirs(results_path, exist_ok=True)
        logger = init_logger(results_path)
        logger.info("[Main] Parsed args:\n%s", args)
        with open(os.path.join(results_path, "params.json"), "w") as fp:
            json.dump(vars(args), fp, indent=2)
        if not os.path.exists(WANDB_API_KEY_FILE):
            raise RuntimeError(f"WANDB API key file not found at {WANDB_API_KEY_FILE}")
        with open(WANDB_API_KEY_FILE) as kf:
            key = json.load(kf)["key"]
        wandb.login(key=key)
        wandb_run = wandb.init(
            project=args.project_name,
            name=experiment_id,
            config=vars(args),
            resume="allow",
        )
    else:
        # ensure path exists before creating a logger
        if is_main_process():
            os.makedirs(results_path, exist_ok=True)
            logger = init_logger(results_path)
        else:
            logger = None
        wandb_run = None

    fp_loader = make_fp_loader(args.fp_type, entropy_out_dim = args.out_dim, retrieval_path=os.path.join(DATASET_ROOT, 'retrieval.pkl'))
    model = SPECTRE(args, fp_loader)
    data_module = SPECTREDataModule(args, fp_loader)

    if args.train:
        train(args, data_module, model, results_path, wandb_run=wandb_run)
    elif args.test:
        test(args, data_module, model, results_path,
             ckpt_path=args.load_from_checkpoint, wandb_run=wandb_run, sweep=True)
    else:
        raise ValueError("[Main] Both --no_train and --no_test set; nothing to do!")

    if is_main_process() and args.train:
        logger and logger.info("[Main] Moving results to final destination")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        shutil.move(results_path, final_path)
        if wandb_run is not None:
            wandb.finish()

if __name__ == "__main__":
    main()
