import argparse
import os
import json

from .const import DO_NOT_OVERRIDE
from .settings import Args

def add_bool_flag(parser: argparse.ArgumentParser, name: str, default: bool):
    if default:
        parser.add_argument(f'--no_{name}', dest=name, action='store_false')
    else:
        parser.add_argument(f'--{name}', dest=name, action='store_true')
    parser.set_defaults(**{name: default})

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_name')
    parser.add_argument('--code_root')
    parser.add_argument('--inference_root')
    parser.add_argument('--data_root')
    parser.add_argument('--split', choices=['train', 'val', 'test'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--load_from_checkpoint')

    parser.add_argument('--input_types', nargs='+', choices=['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mw'])
    parser.add_argument('--requires', nargs='+', choices=['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mw'])

    add_bool_flag(parser, 'debug', False)
    add_bool_flag(parser, 'persistent_workers', True)
    add_bool_flag(parser, 'validate_all', False)
    add_bool_flag(parser, 'use_cached_datasets', True)
    add_bool_flag(parser, 'use_peak_values', False)
    add_bool_flag(parser, 'save_params', True)
    add_bool_flag(parser, 'freeze_weights', False)
    add_bool_flag(parser, 'use_jaccard', False)
    add_bool_flag(parser, 'rank_by_soft_output', True)
    add_bool_flag(parser, 'rank_by_test_set', False)
    add_bool_flag(parser, 'train', True)
    add_bool_flag(parser, 'test', True)

    parser.add_argument('--model_mode', choices=['self', 'cross', 'mixed', 'mixed2'], required=True)
    parser.add_argument('--fp_type', choices=['Entropy', 'HYUN', 'Normal'])
    parser.add_argument('--fp_radius', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--jittering', type=float)

    parser.add_argument('--dim_model', type=int)
    parser.add_argument('--dim_coords', type=int, nargs=3)
    parser.add_argument('--heads', type=int)
    parser.add_argument('--layers', type=int)
    parser.add_argument('--ff_dim', type=int)
    parser.add_argument('--out_dim', type=int)
    parser.add_argument('--accumulate_grad_batches_num', type=int)

    parser.add_argument('--dropout', type=float)
    parser.add_argument('--ranking_set_path')

    parser.add_argument('--lr', type=float)
    parser.add_argument('--noam_factor', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--l1_decay', type=float)
    parser.add_argument('--scheduler', choices=['attention'])
    parser.add_argument('--warm_up_steps', type=int)
    
    add_bool_flag(parser, 'visualize', False)

    args = parser.parse_args()
    model_mode = args.model_mode
    delattr(args, 'model_mode')
    if args.load_from_checkpoint:
        checkpoint_dir = os.path.dirname(args.load_from_checkpoint)
        params_path = os.path.join(checkpoint_dir, 'params.json')
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"No params.json found in checkpoint directory: {params_path}")
        
        with open(params_path, 'r') as f:
            checkpoint_args_dict = json.load(f)

        for k, v in checkpoint_args_dict.items():
            if k in DO_NOT_OVERRIDE:
                continue
            setattr(args, k, v)
    args_dict = {k: v for k, v in vars(args).items() if v is not None}

    return Args(**args_dict), model_mode