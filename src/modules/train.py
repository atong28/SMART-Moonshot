import logging
import torch
import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from .marina import MARINA, MARINAArgs, MARINADataModule
from .spectre import SPECTRE, SPECTREArgs

from .test import test


def train(
    args: MARINAArgs | SPECTREArgs,
    data_module: MARINADataModule,
    model: MARINA | SPECTRE,
    results_path: str,
    wandb_run=None
) -> None:
    """
    Train a MARINA or SPECTRE model using PyTorch Lightning.

    This function configures logging, callbacks, checkpointing, 
    early stopping, learning rate monitoring, and distributed 
    training settings. It initializes a PyTorch Lightning Trainer, 
    runs the full training loop, and optionally executes the test 
    phase after training completes.

    Args:
        args (MARINAArgs | SPECTREArgs):
            Parsed configuration object containing all
            model, training, and optimization hyperparameters.
        data_module (MARINADataModule | SPECTREDataModule):
            The Lightning DataModule responsible for preparing,
            loading, and batching training/validation/test data.
        model (MARINA | SPECTRE):
            The instantiated model to be trained.
        results_path (str):
            Directory path used to store checkpoints, metrics,
            and training artifacts.
        wandb_run (wandb.sdk.wandb_run.Run | None, optional):
            Optional Weights & Biases run object. If provided,
            training metrics will be logged to W&B. Defaults to None.

    Returns:
        None
    """

    torch.set_float32_matmul_precision('high')

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
        save_top_k=1,
        dirpath=results_path,
        filename='epoch_{epoch:d}'
    )

    early_stopping = EarlyStopping(
        monitor=metric,
        mode='max',
        patience=args.patience
    )

    lr_monitor = cb.LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        logger=wandb_logger,
        callbacks=[early_stopping, lr_monitor, ckpt_callback],
        accumulate_grad_batches=args.accumulate_grad_batches_num,
        strategy='ddp',
        gradient_clip_val=1.0
    )

    logger.info("[Main] Begin Training!")

    trainer.fit(model, datamodule=data_module)

    trainer.strategy.barrier()

    if args.test and trainer.local_rank == 0:
        test(args, data_module, model, results_path, None)
