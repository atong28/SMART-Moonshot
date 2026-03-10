from pydantic.dataclasses import dataclass
from dataclasses import field
from ..core import SMARTArgs


@dataclass
class DiffMSArgs(SMARTArgs):
    experiment_name: str = 'diffms-development'
    project_name: str = 'DiffMS'

    log_every_steps: int = 100
    sample_every_val: int = 1000
    samples_to_generate: int = 500
    lambda_train: list[float] = field(default_factory=lambda: [0, 1, 0])
    diffusion_steps: int = 500
    n_layers: int = 5
    hidden_mlp_dims: dict = field(default_factory=lambda: {'X': 256, 'E': 128, 'y': 2048})    
    hidden_dims: dict = field(default_factory=lambda: {
        'dx': 256,
        'de': 64,
        'dy': 1024,
        'n_head': 8,
        'dim_ffX': 256,
        'dim_ffE': 128,
        'dim_ffy': 1024
    })
    
    lr: float = 0.0015
    weight_decay: float = 1e-12
    pct_start: float = 0.3
    
    denoise_nodes: bool = False # TODO: remove?
    diffusion_noise_schedule: str = 'cosine'