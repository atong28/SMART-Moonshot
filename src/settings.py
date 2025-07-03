from typing import Literal, List, Optional
from pydantic.dataclasses import dataclass
from dataclasses import field

@dataclass
class Args:
    experiment_name: str = 'spectre_reproduce'
    code_root: str = '/root/gurusmart/Moonshot'
    inference_root: str = '/root/gurusmart/Moonshot/inference_data'
    data_root: str = '/workspace'
    split: Literal['train', 'val', 'test'] = 'train'
    seed: int = 0
    load_from_checkpoint: str | None = None
    train: bool = True
    test: bool = True
    
    input_types: List[Literal['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mw', 'iso_dist']] = field(
        default_factory=lambda: ['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mw', 'iso_dist']
    )
    requires: List[Literal['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mw', 'iso_dist']] = field(default_factory=lambda: ['hsqc'])
    
    debug: bool = False
    fp_type: Literal['Entropy', 'HYUN', 'Normal'] = 'Entropy'
    fp_radius: Optional[int] = 6
    batch_size: int = 32
    num_workers: int = 8
    epochs: int = 300
    patience: int = 7
    persistent_workers: bool = True
    validate_all: bool = False
    use_cached_datasets: bool = True
    
    jittering: float = 0.0
    use_peak_values: bool = False

    # model args
    dim_model: int = 784
    dim_coords: List[int] = field(default_factory=lambda: [365, 365, 54])
    heads: int = 8
    layers: int = 16
    ff_dim: int = 3072
    out_dim: int = 16384
    accumulate_grad_batches_num: int = 4

    wavelength_bounds: Optional[None] = None
    dropout: float = 0.1
    save_params: bool = True
    ranking_set_path: str = ''

    # training args
    lr: float = 1e-5
    noam_factor: float = 1.0
    weight_decay: float = 0.0
    l1_decay: float = 0.0
    scheduler: Optional[Literal['attention']] = None
    warm_up_steps: int = 0
    freeze_weights: bool = False
    use_jaccard: bool = False

    # testing args
    test_on_deepsat_retrieval_set: bool = False
    rank_by_soft_output: bool = True
    rank_by_test_set: bool = False
