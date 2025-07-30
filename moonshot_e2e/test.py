import os
import pickle
import logging

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from .src.settings import Args
from .src.model import Moonshot
from .src.dataset import MoonshotDataModule

logger = logging.getLogger("lightning")

def test(
    args: Args,
    results_path: str,
    model: Moonshot,
    ckpt_path: str | None = None,
    wandb_run=None
):
    """
    Test the diffusion‐based Moonshot model.

    - args: parsed Args object
    - results_path: directory where checkpoints / outputs are stored
    - model: an (uninitialized) Moonshot instance
    - ckpt_path: optional path to a .ckpt file to load before testing
    - wandb_run: W&B run handle (may be None)
    """
    # 1) Logger & Trainer
    wandb_logger = WandbLogger(experiment=wandb_run)
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=wandb_logger
    )

    # 2) (Re)load best checkpoint if provided
    if ckpt_path is not None:
        logger.info(f"[Test] Loading checkpoint from {ckpt_path}")
        model = Moonshot.load_from_checkpoint(ckpt_path, args=args)

    # 3) DataModule setup
    data_module = MoonshotDataModule(args, results_path)
    data_module.setup(stage="test")

    # 4) Run test
    logger.info("[Test] Starting test()")
    test_results = trainer.test(model, datamodule=data_module)

    # 5) Save results
    out_path = os.path.join(results_path, "test_results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(test_results, f)
    logger.info(f"[Test] Saved test results to {out_path}")

    return test_results
