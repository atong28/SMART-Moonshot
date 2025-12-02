from typing import Literal, Dict, List, Set

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

NON_SPECTRAL_INPUTS: Set[INPUT_TYPES] = set()

if 'nas-gpu' in __file__:
    print('Detected yuzu setup')
    CODE_ROOT = '/data/nas-gpu/wang/atong/SMART-Moonshot'
    DATASET_ROOT = '/data/nas-gpu/wang/atong/MoonshotDatasetv3'
    WANDB_API_KEY_FILE = '/data/nas-gpu/wang/atong/SMART-Moonshot/wandb_api_key.json'
    PVC_ROOT = CODE_ROOT
else:
    print('Detected nautilus setup')
    CODE_ROOT = '/code'
    DATASET_ROOT = '/workspace'
    WANDB_API_KEY_FILE = '/root/gurusmart/Moonshot/wandb_api_key.json'
    PVC_ROOT = '/root/gurusmart/Moonshot'

DO_NOT_OVERRIDE = [
    'train', 'test', 'visualize', 'load_from_checkpoint', 'input_types', 'requires', 'train_lora',
    'lora_rank_qkv', 'lora_rank_out', 'lora_rank_fc', 'lora_scale_qkv', 'lora_scale_out',
    'lora_scale_fc', 'lora_enable_attn', 'lora_enable_fc', 'adapter_dir', 'train_adapter_for_combo',
    'lora_only', 'lora_lr', 'lora_weight_decay', 'full_mix_ratio', 'distill_full_alpha',
    'distill_target'
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