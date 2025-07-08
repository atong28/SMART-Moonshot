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

from src.settings import Args
from src.model import SPECTRE, OptionalInputSPECTRE
from src.dataset import MoonshotDataModule


def test(args: Args, data_module: MoonshotDataModule, results_path: str, model: SPECTRE | OptionalInputSPECTRE, ckpt_path: str | None = None, wandb_run = None):
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
    model.setup_ranker()
    test_result = trainer.test(model, data_module, ckpt_path=ckpt_path)
    if hasattr(model, "mw_rank_records"):
        all_records = []
        for records in model.mw_rank_records.values():
            all_records.extend(records)
        df = pd.DataFrame(all_records)
        df = df[df["mw"].notnull()]  # remove any missing MWs
        print("Global avg:", df[["rank_1", "rank_5", "rank_10"]].mean())

        bins = np.linspace(df["mw"].min(), df["mw"].max(), num=15)
        df["mw_bin"] = pd.cut(df["mw"], bins=bins)
        binned = df.groupby("mw_bin")[["rank_1", "rank_5", "rank_10"]].mean().reset_index()
        binned["bin_center"] = binned["mw_bin"].apply(lambda x: (x.left + x.right) / 2)

        plt.figure(figsize=(10, 6))
        plt.plot(binned["bin_center"], binned["rank_1"], label="Rank-1 Accuracy", marker='o')
        plt.plot(binned["bin_center"], binned["rank_5"], label="Rank-5 Accuracy", marker='o')
        plt.plot(binned["bin_center"], binned["rank_10"], label="Rank-10 Accuracy", marker='o')
        plt.xlabel("Molecular Weight")
        plt.ylabel("Accuracy")
        plt.title("Molecular Weight vs Ranking Accuracy")
        plt.legend()
        plt.grid(True)

        # Save to file
        plot_path = os.path.join(results_path, "mw_vs_rank_accuracy.png")
        plt.savefig(plot_path)
        plt.close()

        # Histogram: sample count per MW bin
        bin_counts = df.groupby("mw_bin").size().reset_index(name="count")
        bin_counts["bin_center"] = bin_counts["mw_bin"].apply(lambda x: (x.left + x.right) / 2)

        plt.figure(figsize=(10, 4))
        plt.bar(bin_counts["bin_center"], bin_counts["count"], width=(bins[1] - bins[0]) * 0.9, align='center')
        plt.xlabel("Molecular Weight")
        plt.ylabel("Sample Count")
        plt.title("Molecular Weight Distribution")
        plt.grid(True)

        hist_path = os.path.join(results_path, "mw_distribution_histogram.png")
        plt.savefig(hist_path)
        plt.close()


    with open(os.path.join(results_path, 'test_result.pkl'), "wb") as f:
        pickle.dump(test_result, f)