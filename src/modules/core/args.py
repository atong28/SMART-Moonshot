from typing import Literal, List, Optional
from pydantic.dataclasses import dataclass
from dataclasses import field


@dataclass
class SMARTArgs:
    experiment_name: str = 'smart-development'
    project_name: str = 'SMART'
    seed: int = 0
    load_from_checkpoint: str | None = None
    train: bool = True
    test: bool = True

    input_types: List[Literal['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mw']] = field(
        default_factory=lambda: ['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mw']
    )

    requires: List[Literal['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mw']] = field(
        default_factory=lambda: []
    )

    debug: bool = False
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 750
    patience: int = 30
    persistent_workers: bool = True

    jittering: float = 1.0

    # training args
    lr: float = 2e-4
    eta_min: float = 1e-5
    weight_decay: float = 0.0
    scheduler: Literal['cosine', 'none'] = 'cosine'
    freeze_weights: bool = False
    use_jaccard: bool = False
    warmup: bool = False
    accumulate_grad_batches_num: int = 4
    dropout: float = 0.1

    visualize: bool = False
    lambda_hybrid: float = 0.0
    fp_type: Literal['RankingEntropy'] = 'RankingEntropy'
    
    drop_me_sign_percentage: float = 0.5
