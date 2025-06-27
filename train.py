import pathlib
import yaml
from datetime import datetime

import logging, os, sys, torch
import random, pickle
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import summarize
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch.distributed as dist

from src.settings import Args
from src.model import build_model
from src.dataset import MoonshotDataModule

def init_logger(path):
    logger = logging.getLogger("lightning")
    logger.setLevel(logging.DEBUG)
    file_path = os.path.join(path, "logs.txt")
    with open(file_path, 'w') as fp: # touch
        pass
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(file_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger

def seed_everything(seed):
    """
    Set the random seed for reproducibility.
    """
    pl.seed_everything(seed,  workers=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)
    # torch.use_deterministic_algorithms(True)

def train(args: Args, overrides: dict | None = None):
    if overrides is not None:
        args = Args(**{**vars(args), **overrides})
    seed_everything(seed=args.seed)
    torch.set_float32_matmul_precision('medium')
    
    optional_inputs = set(args.requires) != set(args.input_types)
    
    if args.debug:
        args.epochs = 10
    
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_path = os.path.join(args.data_root, 'results', args.experiment_name, now)
    while os.path.exists(results_path):
        results_path += '_copy'
    os.makedirs(results_path)
    my_logger = init_logger(results_path)
    my_logger.info(f'[Main] Results Path: {results_path}')
    try:
        my_logger.info(f'[Main] Using GPU : {torch.cuda.get_device_name()}')
    except:
        my_logger.info(f'[Main] Using GPU: unknown type')

    # Trainer, callbacks
    wandb_logger = WandbLogger(name=args.experiment_name + f'_{now}', project="SPECTRE")
    metric = 'val/mean_cos'
    ckpt_callback = cb.ModelCheckpoint(
        monitor=metric,
        mode='max',
        save_last=False,
        save_top_k = 1,
        dirpath=results_path
    )
    early_stopping = EarlyStopping(monitor=metric, mode='max', patience=args.patience)
    lr_monitor = cb.LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        logger=wandb_logger, 
        callbacks=[early_stopping, lr_monitor, ckpt_callback],
        accumulate_grad_batches=args.accumulate_grad_batches_num,
    )

    data_module = MoonshotDataModule(args)
    
    model = build_model(args, optional_inputs, combinations_names=data_module.combinations_names)
    
    if trainer.global_rank == 0:
        my_logger.info(f"[Main] Model Summary: {summarize(model)}")
    
    my_logger.info("[Main] Begin Training!")
    trainer.fit(model, data_module, ckpt_path=args.load_from_checkpoint)

    # Ensure all processes synchronize before switching to test mode
    trainer.strategy.barrier()

if __name__ == '__main__':
    args = Args()
    train(args)