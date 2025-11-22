import argparse
import os
import json
from typing import Union

from .core.const import DO_NOT_OVERRIDE
from .marina import MARINAArgs
from .spectre import SPECTREArgs

def add_bool_flag(parser: argparse.ArgumentParser, name: str, default: bool):
    if default:
        parser.add_argument(f'--no_{name}', dest=name, action='store_false')
    else:
        parser.add_argument(f'--{name}', dest=name, action='store_true')
    parser.set_defaults(**{name: default})


def parse_args() -> Union[MARINAArgs, SPECTREArgs]:
    parser = argparse.ArgumentParser()

    # Architecture selection
    parser.add_argument(
        '--arch',
        choices=['marina', 'spectre'],
        required=True,
        default='marina',
        help='Architecture to use: marina or spectre'
    )

    # Basic configuration
    parser.add_argument('--experiment_name')
    parser.add_argument('--project_name')
    parser.add_argument(
        '--seed',
        type=int
    )
    parser.add_argument('--load_from_checkpoint')

    # Input configuration
    parser.add_argument(
        '--input_types',
        nargs='+',
        choices=[
            'hsqc',
            'c_nmr',
            'h_nmr',
            'mass_spec',
            'mw'
        ]
    )
    parser.add_argument(
        '--requires',
        nargs='+',
        choices=[
            'hsqc',
            'c_nmr',
            'h_nmr',
            'mass_spec',
            'mw'
        ]
    )

    # Boolean flags
    add_bool_flag(parser, 'debug', False)
    add_bool_flag(parser, 'train', True)
    add_bool_flag(parser, 'test', True)
    add_bool_flag(parser, 'persistent_workers', True)
    add_bool_flag(parser, 'use_peak_values', False)
    add_bool_flag(parser, 'freeze_weights', False)
    add_bool_flag(parser, 'use_jaccard', False)
    add_bool_flag(parser, 'warmup', True)
    add_bool_flag(parser, 'visualize', False)
    add_bool_flag(parser, 'hybrid_early_stopping', False)

    # Data loading and training configuration
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--jittering', type=float)

    # Model architecture
    parser.add_argument('--dim_model', type=int)
    parser.add_argument('--heads', type=int)
    parser.add_argument('--layers', type=int)
    parser.add_argument('--self_attn_layers', type=int)
    parser.add_argument('--ff_dim', type=int)
    parser.add_argument('--out_dim', type=int)
    parser.add_argument('--dropout', type=float)

    # Training hyperparameters
    parser.add_argument('--lr', type=float)
    parser.add_argument('--eta_min', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--accumulate_grad_batches_num', type=int)
    parser.add_argument('--scheduler', choices=['cosine', 'none'], default='cosine')

    # Loss and fingerprint configuration
    parser.add_argument("--lambda_hybrid", type=float)
    parser.add_argument("--fp_type", type=str, choices=['RankingEntropy'])

    # MARINA-specific arguments
    args = parser.parse_args()
    # Determine which args class to use based on architecture
    arch = args.arch
    if args.load_from_checkpoint:
        checkpoint_dir = os.path.dirname(args.load_from_checkpoint)
        params_path = os.path.join(checkpoint_dir, 'params.json')
        if not os.path.exists(params_path):
            raise FileNotFoundError(
                f"No params.json found in checkpoint directory: {params_path}")

        with open(params_path, 'r') as f:
            checkpoint_args_dict = json.load(f)

        for k, v in checkpoint_args_dict.items():
            if k in DO_NOT_OVERRIDE or k == 'arch':
                continue
            setattr(args, k, v)
    if args.debug and args.epochs is None:
        args.epochs = 1

    args_dict = {k: v for k, v in vars(
        args).items() if v is not None and k != 'arch'}
    if args_dict['scheduler'] == 'none':
        args_dict['scheduler'] = None
    if arch == 'spectre':
        args_dict.pop('hybrid_early_stopping', None)

    if arch == 'marina':
        return MARINAArgs(**args_dict)
    elif arch == 'spectre':
        return SPECTREArgs(**args_dict)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
