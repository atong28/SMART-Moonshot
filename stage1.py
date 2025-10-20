import logging
import pytorch_lightning as pl
import warnings
import os
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

import numpy as np
import random
import torch
from src.marina.core.args import parse_args
from src.marina.core.settings import MARINAArgs
from src.marina.data.fp_loader import make_fp_loader
from src.marina.arch.model import SPECTRE
from src.marina.train import train
from src.marina.test import test
from src.marina.core.const import DATASET_ROOT, CODE_ROOT, WANDB_API_KEY_FILE
from src.marina.lora.spectre_lora import SPECTRELoRA
from src.marina.lora.load_utils import load_base_ckpt_into_lora_model
from src.marina.data.dataset import SPECTREDataModule

torch.autograd.set_detect_anomaly(True)

def seed_everything(seed):
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    args: MARINAArgs = parse_args()
    seed_everything(args.seed)

    # build a common results path
    today = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_path = os.path.join(DATASET_ROOT, "results", args.experiment_name, today)
    final_path = os.path.join(CODE_ROOT, "results", args.experiment_name, today)
    experiment_id = f"{args.experiment_name}_{today}"

    # rank 0: create outputs, logging, wandb
    if is_main_process() and args.train and not args.visualize:
        os.makedirs(results_path, exist_ok=True)
        logger = init_logger(results_path)
        logger.info("[Main] Parsed args:\n%s", args)

        # dump params.json
        with open(os.path.join(results_path, "params.json"), "w") as fp:
            json.dump(vars(args), fp, indent=2)

        # login to wandb (optional hardening)
        try:
            with open(WANDB_API_KEY_FILE) as kf:
                key = json.load(kf)["key"]
            wandb.login(key=key)
            wandb_run = wandb.init(
                project=args.project_name,
                name=experiment_id,
                config=vars(args),
                resume="allow",
            )
        except FileNotFoundError:
            logger.warning("wandb_api_key.json not found — continuing without W&B.")
            wandb_run = None
    else:
        # ensure path exists before creating a logger
        if is_main_process():
            os.makedirs(results_path, exist_ok=True)
            logger = init_logger(results_path)
        else:
            logger = None
        wandb_run = None

    # ----------------------------
    # Model construction branches
    # ----------------------------
    if args.train_lora:
        assert args.arch == 'v1', 'LoRA not supported for v2'
        if not args.load_from_checkpoint:
            raise ValueError("--train_lora requires --load_from_checkpoint to be set to a base checkpoint")
        fp_loader = make_fp_loader(args.fp_type, entropy_out_dim = args.out_dim)
        model = SPECTRELoRA(args, fp_loader)
        info = load_base_ckpt_into_lora_model(model, args.load_from_checkpoint)
        if is_main_process() and logger:
            logger.info("[LoRA] Loaded base→LoRA: %s", info)
        model.freeze_base_enable_lora()
        if args.lora_lr is not None:
            args.lr = args.lora_lr
        if args.lora_weight_decay is not None:
            args.weight_decay = args.lora_weight_decay
        if not args.train_adapter_for_combo:
            raise ValueError("train_adapter_for_combo is empty; provide modalities like '{hsqc,h_nmr}'.")
        args.input_types = args.train_adapter_for_combo
        args.requires = args.train_adapter_for_combo
        data_module = SPECTREDataModule(args, fp_loader)
    else:
        fp_loader = make_fp_loader(args.fp_type, entropy_out_dim = args.out_dim, retrieval_path=os.path.join(DATASET_ROOT, 'retrieval.pkl'))
        model = SPECTRE(args, fp_loader)
        data_module = SPECTREDataModule(args, fp_loader)

    # ----------------------------
    # Train / Test
    # ----------------------------
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
