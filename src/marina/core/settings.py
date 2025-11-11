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
    requires: List[Literal['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mw']] = field(default_factory=lambda: [])

    debug: bool = False
    batch_size: int = 32
    num_workers: int = 8
    epochs: int = 750
    patience: int = 30
    persistent_workers: bool = True
    
    jittering: float = 1.0
    use_peak_values: bool = False

    # training args
    lr: float = 2e-4
    eta_min: float = 1e-5
    weight_decay: float = 0.0
    scheduler: Optional[Literal['cosine']] = 'cosine'
    freeze_weights: bool = False
    use_jaccard: bool = False
    warmup: bool = True
    accumulate_grad_batches_num: int = 4
    dropout: float = 0.1
    
    visualize: bool = False
    lambda_hybrid: float = 0.0
    fp_type: Literal['RankingEntropy'] = 'RankingEntropy'
    hybrid_early_stopping: bool = False

@dataclass
class MARINAArgs(SMARTArgs):
    experiment_name: str = 'marina-development'
    project_name: str = 'MARINA'
    
    dim_model: int = 784
    nmr_dim_coords: List[int] = field(default_factory=lambda: [365, 365, 54])
    ms_dim_coords: List[int] = field(default_factory=lambda: [392, 392, 0])
    mw_dim_coords: List[int] = field(default_factory=lambda: [784, 0, 0])
    heads: int = 8
    layers: int = 16
    self_attn_layers: int = 2
    ff_dim: int = 3072
    out_dim: int = 16384

    c_wavelength_bounds: List[float] = field(default_factory=lambda: [0.01, 400.0])
    h_wavelength_bounds: List[float] = field(default_factory=lambda: [0.01, 20.0])
    mz_wavelength_bounds: List[float] = field(default_factory=lambda: [0.01, 5000.0])
    intensity_wavelength_bounds: List[float] = field(default_factory=lambda: [0.001, 200.0])
    

@dataclass
class SPECTREArgs(SMARTArgs):
    experiment_name: str = 'spectre-development'
    project_name: str = 'SPECTRE'
    
    dim_model: int = 784
    nmr_dim_coords: List[int] = field(default_factory=lambda: [365, 365, 54])
    ms_dim_coords: List[int] = field(default_factory=lambda: [392, 392, 0])
    mw_dim_coords: List[int] = field(default_factory=lambda: [784, 0, 0])
    heads: int = 8
    layers: int = 16
    self_attn_layers: int = 2
    ff_dim: int = 3072
    out_dim: int = 16384
    
    c_wavelength_bounds: List[float] = field(default_factory=lambda: [0.01, 400.0])
    h_wavelength_bounds: List[float] = field(default_factory=lambda: [0.01, 20.0])
    mz_wavelength_bounds: List[float] = field(default_factory=lambda: [0.01, 5000.0])
    intensity_wavelength_bounds: List[float] = field(default_factory=lambda: [0.001, 200.0])
    mw_wavelength_bounds: List[float] = field(default_factory=lambda: [0.01, 7000.0])