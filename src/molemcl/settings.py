# molemcl/settings.py

from typing import Literal, Optional
from pydantic.dataclasses import dataclass

@dataclass
class MoleMCLArgs:
    # run id
    experiment_name: str = "molemcl-development"
    project_name: str = "MoleMCL"
    seed: int = 0
    train: bool = True
    test: bool = False
    load_from_checkpoint: Optional[str] = None  # for test/inference

    # data
    ae_root: str = "/data/nas-gpu/wang/atong/MoonshotDataset"
    train_on_duplicates: bool = False
    batch_size: int = 256
    num_workers: int = 8
    persistent_workers: bool = True
    pin_memory: bool = True
    mask_rate: float = 0.15
    mask_edge: bool = True

    # model
    num_layer: int = 10
    emb_dim: int = 784
    gnn_type: Literal["gin", "gcn", "gat", "graphsage"] = "gin"
    JK: Literal["last", "sum", "max", "concat"] = "last"
    drop_ratio: float = 0.0
    alpha: float = 0.5
    temperature: float = 0.1

    # optim
    lr: float = 1e-3
    weight_decay: float = 0.0

    # trainer
    max_epochs: int = 100
    accelerator: Literal["auto", "cpu", "gpu"] = "gpu"
    devices: int = 1
    log_every_n_steps: int = 50
    deterministic: bool = False

    # I/O
    ckpt_path: Optional[str] = None
    output_dir: str = "./results"   # base folder like Stage 1

    # early stopping
    early_stopping: bool = True
    early_stopping_monitor: str = "val/loss"
    early_stopping_mode: str = "min"   # "min" for loss, "max" for metrics
    early_stopping_patience: int = 10  # epochs with no improvement
    early_stopping_min_delta: float = 0.0