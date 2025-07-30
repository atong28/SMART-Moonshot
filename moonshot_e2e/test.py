import os
import pickle

import wandb
import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
from pytorch_lightning.loggers import WandbLogger

from .src.settings import Args
from .src.model import MoonshotDiffusion
from .src.dataset import MoonshotDataModule

def test(
    args: Args,
    data_module: MoonshotDataModule,
    model: MoonshotDiffusion,
    results_path: str,
    ckpt_path: str | None = None,
    wandb_run = None
):
    # initialize WandB (if provided)
    wandb_logger = WandbLogger(experiment=wandb_run)

    # we don't need early stopping / checkpointing during test,
    # but you can re-use callbacks array if you like
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        strategy="ddp_find_unused_parameters_true",
        logger=wandb_logger,
    )

    # run the test loop; this will call your model.test_step / on_test_epoch_end
    test_results = trainer.test(
        model,
        datamodule=data_module,
        ckpt_path = ckpt_path
    )

    # save raw test outputs
    out_file = os.path.join(results_path, "test_results.pkl")
    with open(out_file, "wb") as f:
        pickle.dump(test_results, f)

    return test_results
