from typing import Literal, List, Optional
from pydantic.dataclasses import dataclass
from dataclasses import field

from ..core import SMARTArgs


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

    c_wavelength_bounds: List[float] = field(
        default_factory=lambda: [0.01, 400.0])
    h_wavelength_bounds: List[float] = field(
        default_factory=lambda: [0.01, 20.0])
    mz_wavelength_bounds: List[float] = field(
        default_factory=lambda: [0.01, 5000.0])
    intensity_wavelength_bounds: List[float] = field(
        default_factory=lambda: [0.001, 200.0])
