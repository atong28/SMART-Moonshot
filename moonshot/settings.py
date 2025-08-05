from typing import Literal, List, Dict, Optional
from pydantic.dataclasses import dataclass
from dataclasses import field
from pathlib import Path

@dataclass
class SpectreArgs:
    experiment_name: str = 'mixed-attention-development'
    project_name: str = 'SPECTRE'
    code_root: str = str(Path(__file__).resolve().parent)
    inference_root: str = str(Path(__file__).resolve().parent.parent / "inference_data")
    data_root: str = '/data/nas-gpu/wang/atong/MoonshotDataset'
    split: Literal['train', 'val', 'test'] = 'train'
    seed: int = 0
    load_from_checkpoint: str | None = None
    train: bool = True
    test: bool = True
    
    input_types: List[Literal['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mw', 'formula']] = field(
        default_factory=lambda: ['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mw', 'formula']
    )
    requires: List[Literal['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mw', 'formula']] = field(default_factory=lambda: [])
    
    debug: bool = False
    fp_type: Literal['Entropy', 'HYUN', 'Normal'] = 'Entropy'
    fp_radius: Optional[int] = 6
    batch_size: int = 32
    num_workers: int = 8
    epochs: int = 500
    patience: int = 30
    persistent_workers: bool = True
    use_cached_datasets: bool = False
    
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
    id_wavelength_bounds: List[float] = field(default_factory=lambda: [0.01, 7000.0])
    abundance_wavelength_bounds: List[float] = field(default_factory=lambda: [0.0001, 2.0])
    mw_wavelength_bounds: List[float] = field(default_factory=lambda: [0.0001, 7000.0])
    dropout: float = 0.1
    save_params: bool = True
    ranking_set_path: str = ''

    # training args
    lr: float = 5e-5
    noam_factor: float = 1.0
    weight_decay: float = 0.0
    l1_decay: float = 0.0
    scheduler: Optional[Literal['attention']] = None
    warm_up_steps: int = 0
    freeze_weights: bool = False
    use_jaccard: bool = False

    # testing args
    rank_by_soft_output: bool = True
    rank_by_test_set: bool = False

    visualize: bool = False

@dataclass
class MoonshotArgs:
    # general
    name: str = 'moonshot-devel' # Warning: 'debug' and 'test' are reserved name that have a special behavior
    parent_dir: str = '.'
    wandb: Literal['online', 'offline', 'disabled'] = 'online'
    wandb_name: str = 'Moonshot'
    gpus: int = 1                     
    decoder: str = None # path to pretrained decoder
    encoder: str = None # path to pretrained encoder
    resume: str = None
    test_only: str = None
    load_weights: str = None
    encoder_finetune_strategy: Optional[Literal['freeze', 'ft-unfold', 'freeze-unfold', 'freeze-transformer', 'ft-transformer']] = None
    decoder_finetune_strategy: Optional[Literal['freeze', 'ft-input', 'freeze-input', 'ft-transformer', 'freeze-transformer', 'ft-output']] = None 
    check_val_every_n_epochs: int = 1
    sample_every_val: int = 1000
    val_samples_to_generate: int = 100
    test_samples_to_generate: int = 100
    log_every_steps: int = 50
    evaluate_all_checkpoints: bool = False
    checkpoint_strategy: str = 'last'
    spectre_ckpt: Optional[str] = '/data/nas-gpu/wang/atong/SMART-Moonshot/spectre_ckpt'
    
    # model
    transition: Literal['marginal', 'uniform'] = 'marginal' 
    model: Literal['graph_tf'] = 'graph_tf'
    diffusion_steps: int = 500
    diffusion_noise_schedule: Literal['cosine'] = 'cosine'         
    n_layers: int = 5
    extra_features: Optional[Literal['all', 'cycles', 'eigenvalues']] = 'all'
    hidden_mlp_dims: Dict[str, int] = field(default_factory=lambda: {'X': 256, 'E': 128, 'y': 2048})        
    hidden_dims : Dict[str, int] = field(default_factory=lambda: {'dx': 256, 'de': 64, 'dy': 1024, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 128, 'dim_ffy': 1024})
    encoder_hidden_dim: int = 512          # Large Model Default (MSG)
    encoder_magma_modulo: int = 2048       # Large Model Default (MSG)
    lambda_train: List[int] = field(default_factory=lambda: [0, 1, 0])
    spectre_ckpt: str = '/data/nas-gpu/wang/atong/SMART-Moonshot/spectre_ckpt'
    denoise_nodes: bool = False
    
    # train
    n_epochs: int = 75
    batch_size: int = 32
    eval_batch_size: int = 128
    lr: float = 0.0015 # 0.0015 for training, 0.0002 for fine-tuning
    clip_grad: Optional[float] = None          # float, null to disable
    save_model: bool = True
    num_workers: int = 8
    pin_memory: bool = True
    progress_bar: bool = False
    weight_decay: float = 1e-12
    optimizer: Literal['adamw', 'nadamw', 'nadam'] = 'adamw'
    scheduler: Literal['one_cycle', 'const'] = 'one_cycle' # 'const' | 'one_cycle'
    pct_start: float = 0.3
    seed: int = 0
