from typing import Literal, Dict, List, Set
from pathlib import Path

from ..log import get_logger

logger = get_logger(__file__)

INPUT_TYPES = Literal['hsqc', 'h_nmr', 'c_nmr', 'mass_spec', 'mw']
INPUTS_CANONICAL_ORDER: List[INPUT_TYPES] = ['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mw']

DEBUG_LEN: int = 3000

DROP_PERCENTAGE: Dict[INPUT_TYPES, float] = {
    'hsqc': 0.20990,
    'h_nmr': 0.1604,
    'c_nmr': 0.1604,
    'mass_spec': 0.5,
    'mw': 0.5
}

NON_SPECTRAL_INPUTS: Set[INPUT_TYPES] = {'mw'}
SELF_ATTN_INPUTS: Set[INPUT_TYPES] = {'hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mw'}
if 'src/marina/src/modules' in __file__:
    logger.info('Detected website setup')
    CODE_ROOT = None
    DATASET_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent / 'data'
    WANDB_API_KEY_FILE = None
    PVC_ROOT = None
    BENCHMARK_ROOT = None
elif 'nas-gpu' in __file__:
    logger.info('Detected yuzu setup')
    CODE_ROOT = '/data/nas-gpu/wang/atong/SMART-Moonshot'
    DATASET_ROOT = '/data/nas-gpu/wang/atong/Datasets/MARINAControl1'
    WANDB_API_KEY_FILE = '/data/nas-gpu/wang/atong/SMART-Moonshot/wandb_api_key.json'
    PVC_ROOT = CODE_ROOT
    BENCHMARK_ROOT = '/data/nas-gpu/wang/atong/Datasets/Benchmark'
elif '/code' in __file__:
    logger.info('Detected nautilus setup')
    CODE_ROOT = '/code'
    DATASET_ROOT = '/workspace'
    WANDB_API_KEY_FILE = '/root/gurusmart/Moonshot/wandb_api_key.json'
    PVC_ROOT = '/root/gurusmart/Moonshot'
    BENCHMARK_ROOT = '/root/gurusmart/Benchmark'
else:
    raise ValueError('Unknown setup')

DO_NOT_OVERRIDE = [
    'train', 'test', 'visualize', 'load_from_checkpoint', 'input_types', 'requires',
    'benchmark', 'restrictions'
]

HSQC_TYPE = 0
C_NMR_TYPE = 1
H_NMR_TYPE = 2
MW_TYPE = 3
MS_TYPE = 4

INPUT_MAP = {
    'hsqc': HSQC_TYPE,
    'c_nmr': C_NMR_TYPE,
    'h_nmr': H_NMR_TYPE,
    'mw': MW_TYPE,
    'mass_spec': MS_TYPE
}