import logging
import os
import torch
import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
import sys

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import summarize
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from .src.settings import Args
from .src.model import Moonshot            # your diffusion model
from .src.dataset import MoonshotDataModule  # the new data module
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
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(file_path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger

def train(
    args: Args,
    data_module: MoonshotDataModule,
    model: Moonshot,
    results_path: str,
    wandb_run=None
):
    # ensure consistent precision if desired
    torch.set_float32_matmul_precision('medium')

    # Debug mode: run just one epoch if requested
    if args.debug:
        args.epochs = 1

    # Initialize logging
    logger = init_logger(results_path)
    logger.info(f"[Main] Results Path: {results_path}")
    try:
        logger.info(f"[Main] Using GPU: {torch.cuda.get_device_name()}")
    except:
        logger.info("[Main] Using GPU: unknown")

    # WandB logger
    wandb_logger = WandbLogger(experiment=wandb_run)

    # Callbacks: checkpointing on val/loss (minimize), early stopping, LR monitor
    checkpoint_cb = cb.ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        dirpath=results_path,
        filename="epoch_{epoch:02d}-val_loss_{val_loss:.4f}"
    )
    early_stop = EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=args.patience
    )
    lr_monitor = cb.LearningRateMonitor(logging_interval="step")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        strategy="ddp_find_unused_parameters_true",
        accumulate_grad_batches=args.accumulate_grad_batches_num,
        gradient_clip_val=1.0,
        logger=wandb_logger,
        callbacks=[early_stop, lr_monitor, checkpoint_cb]
    )

    # Log model summary
    logger.info(f"[Main] Model Summary:\n{summarize(model)}")
    logger.info("[Main] Starting training!")

    # Fit
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=args.load_from_checkpoint
    )

    # Synchronize before testing
    trainer.strategy.barrier()

    # Final test if requested
    if args.test and trainer.global_rank == 0:
        logger.info("[Main] Running final test")
        test(
            args=args,
            results_path=results_path,
            model=model,
            ckpt_path=None,
            wandb_run=wandb_run
        )
