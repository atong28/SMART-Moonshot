import os
import pickle

import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning.callbacks as cb
import pytorch_lightning as pl

from ..settings import SPECTREArgs
from .model import SPECTRE
from ..dataset.spectre import SPECTREDataModule


def test(args: SPECTREArgs, data_module: SPECTREDataModule, model: SPECTRE, results_path: str, ckpt_path: str | None = None, wandb_run = None):
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
        accelerator="auto",
        devices=1,
        strategy='auto',
        logger=wandb_logger,
        accumulate_grad_batches=args.accumulate_grad_batches_num,
        callbacks=[early_stopping, lr_monitor, ckpt_callback]
    )
    test_result = trainer.test(model, data_module, ckpt_path=ckpt_path)
    with open(os.path.join(results_path, 'test_result.pkl'), "wb") as f:
        pickle.dump(test_result, f)