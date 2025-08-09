# molemcl/train.py
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from typing import Optional
from .settings import MoleMCLArgs
from ..dataset.molemcl import AEDataModule
from .model import MoleMCLModule

def train(args: MoleMCLArgs, pl_logger: Optional[pl.loggers.Logger] = None):
    dm = AEDataModule(
        ae_root=args.ae_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory,
        mask_rate=args.mask_rate,
        mask_edge=args.mask_edge,
        train_on_duplicates=args.train_on_duplicates,
    )

    model = MoleMCLModule(
        num_layer=args.num_layer,
        emb_dim=args.emb_dim,
        gnn_type=args.gnn_type,
        JK=args.JK,
        drop_ratio=args.drop_ratio,
        alpha=args.alpha,
        temperature=args.temperature,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="molemcl-{epoch:03d}-{val_loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")

    callbacks = [ckpt_cb, lr_cb]
    if args.early_stopping:
        es_cb = EarlyStopping(
            monitor=args.early_stopping_monitor,
            mode=args.early_stopping_mode,
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
            verbose=True,
        )
        callbacks.append(es_cb)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        log_every_n_steps=args.log_every_n_steps,
        deterministic=args.deterministic,
        callbacks=callbacks,
        logger=pl_logger,
    )

    trainer.fit(model, dm, ckpt_path=args.ckpt_path)
