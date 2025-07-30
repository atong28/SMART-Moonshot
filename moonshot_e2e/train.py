# train.py
import logging
import os
import torch
import random
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
import sys

import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import summarize
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from .src.settings import Args
from .src.model import MoonshotDiffusion   # your diffusion LightningModule
from .src.dataset import MoonshotDataModule  # returns (batch_inputs, graph)
from .test import test

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
        with open(file_path, 'w'):
            pass
        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(file_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger

def train(
    args: Args,
    data_module: MoonshotDataModule,
    model: MoonshotDiffusion,
    results_path: str,
    wandb_run = None
):
    # if you want to force reproducibility in matmuls:
    torch.set_float32_matmul_precision('medium')

    # quick debug override
    if args.debug:
        args.epochs = 1

    logger = init_logger(results_path)
    logger.info(f"[Main] Results Path: {results_path}")
    try:
        logger.info(f"[Main] Using GPU: {torch.cuda.get_device_name()}")
    except:
        logger.info("[Main] Using GPU: unknown")

    # --- load pretrained SPECTRE into diffusion backbone ---
    if not args.spectre_ckpt:
        raise ValueError("Please specify --spectre_ckpt to load pretrained SPECTRE backbone")
    logger.info(f"[Main] Loading SPECTRE checkpoint from {args.spectre_ckpt}")
    ckpt = torch.load(args.spectre_ckpt, map_location='cpu')
    # assume checkpoint has state_dict with keys like 'encoder.<whatever>'
    state_dict = ckpt.get('state_dict', ckpt)
    encoder_sd = {}
    for k,v in state_dict.items():
        if k.startswith('encoder.'):
            new_key = k.replace('encoder.', '')
            encoder_sd[new_key] = v
    # load & freeze
    model.backbone.load_state_dict(encoder_sd, strict=True)
    model.backbone.eval()
    for p in model.backbone.parameters():
        p.requires_grad = False

    # --- WandB + callbacks ---
    wandb_logger = WandbLogger(experiment=wandb_run)

    # monitor val/loss (diffusion MSE)
    ckpt_cb = cb.ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        dirpath=results_path,
        filename="epoch{epoch:02d}-val_loss{val_loss:.4f}"
    )
    early_stop = EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=args.patience
    )
    lr_monitor = cb.LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        strategy="ddp_find_unused_parameters_true",
        accumulate_grad_batches=args.accumulate_grad_batches_num,
        gradient_clip_val=1.0,
        logger=wandb_logger,
        callbacks=[early_stop, lr_monitor, ckpt_cb],
        fast_dev_run=args.debug,
    )

    logger.info(f"[Main] Model Summary:\n{summarize(model)}")
    logger.info("[Main] Beginning diffusion training!")
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=args.load_from_checkpoint
    )

    # barrier before test
    trainer.strategy.barrier()

    if args.test and trainer.global_rank == 0:
        logger.info("[Main] Running final test")
        test(args, data_module, results_path, model, ckpt_path=None)
