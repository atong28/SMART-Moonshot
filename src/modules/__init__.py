from .marina import MARINA, MARINAArgs, MARINADataModule
from .spectre import SPECTRE, SPECTREArgs, SPECTREDataModule
from .args import parse_args
from .train import train
from .test import test
from .benchmark import benchmark
__all__ = [
    'MARINA',
    'MARINAArgs',
    'MARINADataModule',
    'SPECTRE',
    'SPECTREArgs',
    'SPECTREDataModule',

    'parse_args',
    'train',
    'test',
    'benchmark'
]
