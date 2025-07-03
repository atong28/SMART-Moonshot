import logging, os, torch
import random
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
import sys

import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import summarize
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import torch.distributed as dist

from src.settings import Args
from src.model import SPECTRE, OptionalInputSPECTRE
from src.dataset import MoonshotDataModule

from test import test

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

def seed_everything(seed):
    """
    Set the random seed for reproducibility.
    """
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)

def train(args: Args, data_module: MoonshotDataModule, model: SPECTRE | OptionalInputSPECTRE, results_path: str, wandb_run = None):
    seed_everything(seed=args.seed)
    torch.set_float32_matmul_precision('medium')

    if args.debug:
        args.epochs = 1
    
    logger = init_logger(results_path)
    logger.info(f'[Main] Results Path: {results_path}')
    try:
        logger.info(f'[Main] Using GPU : {torch.cuda.get_device_name()}')
    except:
        logger.info(f'[Main] Using GPU: unknown type')

    wandb_logger = WandbLogger(experiment=wandb_run)
    metric = 'val/mean_cos'
    ckpt_callback = cb.ModelCheckpoint(
        monitor=metric,
        mode='max',
        save_last=False,
        save_top_k = 1,
        dirpath=results_path,
        filename='epoch_{epoch:d}'
    )
    early_stopping = EarlyStopping(monitor=metric, mode='max', patience=args.patience)
    lr_monitor = cb.LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        logger=wandb_logger, 
        callbacks=[early_stopping, lr_monitor, ckpt_callback],
        accumulate_grad_batches=args.accumulate_grad_batches_num,
        strategy='ddp'
    )

    logger.info(f"[Main] Model Summary: {summarize(model)}")
    logger.info("[Main] Begin Training!")
    trainer.fit(model, data_module, ckpt_path=args.load_from_checkpoint)

    # Ensure all processes synchronize before switching to test mode
    trainer.strategy.barrier()

    if args.test and trainer.local_rank == 0:
        test(args, data_module, results_path, model, None)