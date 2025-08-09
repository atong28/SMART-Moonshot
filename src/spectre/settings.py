from typing import Literal, List, Optional
from pydantic.dataclasses import dataclass
from dataclasses import field

@dataclass
class SPECTREArgs:
    experiment_name: str = 'spectre-development'
    project_name: str = 'SPECTRE'
    seed: int = 0
    load_from_checkpoint: str | None = None
    train: bool = True
    test: bool = True
    
    input_types: List[Literal['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mw', 'formula']] = field(
        default_factory=lambda: ['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mw', 'formula']
    )
    requires: List[Literal['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mw', 'formula']] = field(default_factory=lambda: [])
    
    debug: bool = False
    batch_size: int = 32
    num_workers: int = 8
    epochs: int = 500
    patience: int = 30
    persistent_workers: bool = True
    
    jittering: float = 1.0
    use_peak_values: bool = False

    # model args
    dim_model: int = 784
    nmr_dim_coords: List[int] = field(default_factory=lambda: [365, 365, 54])
    ms_dim_coords: List[int] = field(default_factory=lambda: [392, 392, 0])
    mw_dim_coords: List[int] = field(default_factory=lambda: [784, 0, 0])
    heads: int = 8
    layers: int = 16
    self_attn_layers: int = 2
    ff_dim: int = 3072
    out_dim: int = 16384
    accumulate_grad_batches_num: int = 4

    c_wavelength_bounds: List[float] = field(default_factory=lambda: [0.01, 400.0])
    h_wavelength_bounds: List[float] = field(default_factory=lambda: [0.01, 20.0])
    mz_wavelength_bounds: List[float] = field(default_factory=lambda: [0.01, 5000.0])
    intensity_wavelength_bounds: List[float] = field(default_factory=lambda: [0.001, 200.0])
    mw_wavelength_bounds: List[float] = field(default_factory=lambda: [0.0001, 7000.0])
    dropout: float = 0.1
    save_params: bool = True

    # training args
    lr: float = 5e-5
    noam_factor: float = 1.0
    weight_decay: float = 0.0
    l1_decay: float = 0.0
    scheduler: Optional[Literal['attention']] = None
    warm_up_steps: int = 0
    freeze_weights: bool = False
    use_jaccard: bool = False

    visualize: bool = False