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
from src.marina.core.args import parse_args
from src.marina.core.settings import MARINAArgs
from src.marina.data.fp_loader import EntropyFPLoader
from src.marina.arch.model import SPECTRE
from src.marina.train import train
from src.marina.test import test
from src.marina.core.const import DATASET_ROOT, CODE_ROOT
from src.marina.lora.spectre_lora import SPECTRELoRA
from src.marina.lora.load_utils import load_base_ckpt_into_lora_model
from src.marina.data.dataset import SPECTREDataModule

def seed_everything(seed):
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

args: MARINAArgs = parse_args()
seed_everything(args.seed)

# build a common results path
today = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_path = os.path.join(DATASET_ROOT, "results", args.experiment_name, today)
final_path = os.path.join(CODE_ROOT, "results", args.experiment_name, today)
experiment_id = f"{args.experiment_name}_{today}"

wandb_run = None
logger = None

fp_loader = EntropyFPLoader()
fp_loader.setup(args.out_dim, 6)
data_module = SPECTREDataModule(args, fp_loader)
data_module.setup('fit')
train_dl = data_module.train_dataloader()

def compute_pos_weight_from_loader(loader, device="cpu", eps=1e-6):
    """
    Expects each batch to include `batch_fps` as a float/bool tensor of shape (B, 16384)
    with 1 for active, 0 for inactive.
    """
    pos_counts = 0
    total = 0

    for batch in loader:
        # adjust indexing to how your loader returns items
        # e.g., (inputs, batch_fps) or (something, batch_fps, something_else)
        batch_fps = batch[1] if isinstance(batch, (list, tuple)) else batch["fps"]
        batch_fps = batch_fps.to(device)

        
        pos_counts += batch_fps.sum(dtype=torch.float64)
        total += batch_fps.size(0)

    # negatives per bit
    return pos_counts / total

pos_weight = compute_pos_weight_from_loader(train_dl)
print(pos_weight)