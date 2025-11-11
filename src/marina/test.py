import os
import pickle
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning.callbacks as cb
import pytorch_lightning as pl

from .core.settings import MARINAArgs, SPECTREArgs
from .arch.marina import MARINA
from .arch.spectre import SPECTRE
from .data.marina import MARINADataModule


def test(
    args: MARINAArgs | SPECTREArgs,
    data_module: MARINADataModule,
    model: MARINA | SPECTRE,
    results_path: str,
    ckpt_path: str | None = None,
    wandb_run = None,
    sweep=False
) -> dict:
    if not os.path.exists(results_path):
        os.makedirs(results_path, exist_ok=True)
    model.setup_ranker()
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
        
    result = test_result[0] if isinstance(test_result, list) and test_result and isinstance(test_result[0], dict) else {}

    if sweep:
        return result