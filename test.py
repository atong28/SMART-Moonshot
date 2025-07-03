import os
import pickle

import wandb
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

from src.settings import Args
from src.model import SPECTRE, OptionalInputSPECTRE
from src.dataset import MoonshotDataModule


def test(args: Args, data_module: MoonshotDataModule, results_path: str, model: SPECTRE | OptionalInputSPECTRE | None = None, ckpt_path: str | None = None, wandb_run = None):
    wandb_logger = WandbLogger(experiment=wandb_run)
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,           # force single-GPU
        strategy='auto',
        logger=wandb_logger,
        accumulate_grad_batches=args.accumulate_grad_batches_num
    )
    model.setup_ranker()
    test_result = trainer.test(model, data_module, ckpt_path=ckpt_path)
    with open(os.path.join(results_path, 'test_result.pkl'), "wb") as f:
        pickle.dump(test_result, f)