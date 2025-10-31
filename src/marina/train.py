import logging
import torch
import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from .core.settings import MARINAArgs
from .data.dataset import SPECTREDataModule
from .arch.model import SPECTRE
from .test import test

def train(args: MARINAArgs, data_module: SPECTREDataModule, model: SPECTRE, results_path: str, wandb_run = None):
    torch.set_float32_matmul_precision('high')
    if args.debug:
        args.epochs = 5
    logger = logging.getLogger('lightning')
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
        strategy='ddp_find_unused_parameters_true',
        gradient_clip_val=1.0
    )
    logger.info("[Main] Begin Training!")
    trainer.fit(model, datamodule = data_module)

    trainer.strategy.barrier()

    if args.test and trainer.local_rank == 0:
        test(args, data_module, model, results_path, None)