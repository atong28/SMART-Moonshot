from typing import Literal, List, Optional
from pydantic.dataclasses import dataclass
from dataclasses import field

@dataclass
class MARINAArgs:
    experiment_name: str = 'marina-development'
    project_name: str = 'MARINA'
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
    dropout: float = 0.1
    save_params: bool = True
    
    hybrid_early_stopping: bool = False
    modality_dropout_scheduler: Optional[str] = None # None, 'constant', 'scheduled'

    # training args
    lr: float = 2e-4
    eta_min: float = 1e-5
    noam_factor: float = 1.0
    weight_decay: float = 0.0
    l1_decay: float = 0.0
    scheduler: Optional[Literal['attention', 'cosine']] = 'cosine'
    warm_up_steps: int = 0
    freeze_weights: bool = False
    use_jaccard: bool = False

    visualize: bool = False
    
    # LoRA core
    train_lora: bool = False
    lora_rank_qkv: int = 8
    lora_rank_out: int = 8
    lora_rank_fc: int = 4
    lora_scale_qkv: float = 1.0
    lora_scale_out: float = 1.0
    lora_scale_fc: float = 1.0
    lora_enable_attn: bool = True
    lora_enable_fc: bool = True
    lora_enable_self_attn: bool = False

    # Adapter training
    adapter_dir: str = "adapters"
    train_adapter_for_combo: List[Literal['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mw']] = field(
        default_factory=lambda: ['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mw']
    )
    lora_only: bool = True
    lora_lr: float = 1e-3
    lora_weight_decay: float = 0.0
    full_mix_ratio: float = 0.2
    distill_full_alpha: float = 0.1
    distill_target: str = "logits"  # or "embedding"

    # hybrid loss
    lambda_hybrid: float = 0.0

    # fingerprint type
    fp_type: Literal['RankingEntropy'] = 'RankingEntropy'
