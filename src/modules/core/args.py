from typing import Literal, List, Optional
from pydantic.dataclasses import dataclass
from dataclasses import field


@dataclass
class SMARTArgs:
    experiment_name: str = 'smart-development'
    project_name: str = 'SMART'
    # random seed
    seed: int = 0
    # path to load checkpoint from
    load_from_checkpoint: str | None = None
    # whether to do training
    train: bool = True
    # whether to do testing
    test: bool = True
    # whether to do benchmarking
    benchmark: bool = True
    # restrictions on the input types to be used for benchmarking
    restrictions: Optional[List[Literal['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mw']]] = None

    input_types: List[Literal['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mw']] = field(
        default_factory=lambda: ['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mw']
    )

    requires: List[Literal['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mw']] = field(
        default_factory=lambda: []
    )

    # training args
    debug: bool = False
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 750
    patience: int = 30
    persistent_workers: bool = True
    lr: float = 2e-4
    eta_min: float = 1e-5
    weight_decay: float = 0.0
    scheduler: Literal['cosine', 'none'] = 'cosine'
    freeze_weights: bool = False
    use_jaccard: bool = False
    warmup: bool = False
    accumulate_grad_batches_num: int = 4
    dropout: float = 0.1
    
    # jittering default value to wobble the spectra
    jittering: float = 1.0

    # BCE and cosine similarity loss lambda. 0 for full cosine similarity loss, 1 for full BCE loss.
    lambda_hybrid: float = 0.0
    
    # fp type for prediction and evaluation. fingerprint details should be stored in 
    #   DATASET_ROOT/RankingEntropy/
    # with the proper formatting.
    fp_type: Literal['RankingEntropy'] = 'RankingEntropy'
    
    # additional test types to be used for testing, always will test on all inputs
    additional_test_types: list[list[str]] = field(default_factory=lambda: [
        ['hsqc'], ['h_nmr'], ['c_nmr'], ['mass_spec']
    ])

    # split to use for benchmarking
    benchmark_split: Literal['val', 'test'] = 'val'