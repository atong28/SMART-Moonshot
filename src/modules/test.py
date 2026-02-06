import os
import pickle

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning.callbacks as cb
import pytorch_lightning as pl

from .marina import MARINA, MARINAArgs, MARINADataModule
from .spectre import SPECTRE, SPECTREArgs
from .benchmark import benchmark
from .data.fp_loader import EntropyFPLoader


def test(
    args: MARINAArgs | SPECTREArgs,
    data_module: MARINADataModule,
    model: MARINA | SPECTRE,
    results_path: str,
    ckpt_path: str | None = None,
    wandb_run=None,
    fp_loader: EntropyFPLoader | None = None
) -> dict:
    """
    Run evaluation on a trained MARINA or SPECTRE model.

    This function sets up a PyTorch Lightning Trainer configured for 
    evaluation, restores a checkpoint if provided, and runs the test 
    loop using the given model and DataModule. It logs metrics to 
    Weights & Biases (if enabled) and saves the full test results to 
    a pickle file in the specified results directory.

    Args:
        args (MARINAArgs | SPECTREArgs):
            Parsed arguments containing model and evaluation settings.
        data_module (MARINADataModule | SPECTREDataModule):
            Lightning DataModule providing test data.
        model (MARINA | SPECTRE):
            The model instance to evaluate.
        results_path (str):
            Directory path where evaluation outputs and metrics will be saved.
        ckpt_path (str | None, optional):
            Path to a model checkpoint to load before testing.
            If None, the current model weights are used. Defaults to None.
        wandb_run (wandb.sdk.wandb_run.Run | None, optional):
            Optional Weights & Biases run for logging metrics. Defaults to None.
        sweep (bool, optional):
            If True, return the results dictionary directly for use in W&B
            hyperparameter sweeps. Defaults to False.

    Returns:
        dict:
            A dictionary of test metrics produced by PyTorch Lightning. 
            Returned only when `sweep=True`; otherwise the function returns None.
    """

    if not os.path.exists(results_path):
        os.makedirs(results_path, exist_ok=True)

    model.setup_ranker()

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
        
    if args.benchmark:
        benchmark(args, data_module, model, fp_loader, wandb_run=wandb_run)
