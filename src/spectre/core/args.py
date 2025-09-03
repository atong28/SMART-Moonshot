import argparse
import os
import json

from .const import DO_NOT_OVERRIDE
from .settings import SPECTREArgs

def add_bool_flag(parser: argparse.ArgumentParser, name: str, default: bool):
    if default:
        parser.add_argument(f'--no_{name}', dest=name, action='store_false')
    else:
        parser.add_argument(f'--{name}', dest=name, action='store_true')
    parser.set_defaults(**{name: default})

def parse_args() -> SPECTREArgs:
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_name')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--load_from_checkpoint')

    parser.add_argument('--input_types', nargs='+', choices=['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mw', 'formula'])
    parser.add_argument('--requires', nargs='+', choices=['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mw', 'formula'])

    add_bool_flag(parser, 'debug', False)
    add_bool_flag(parser, 'persistent_workers', True)
    add_bool_flag(parser, 'freeze_weights', False)
    add_bool_flag(parser, 'train', True)
    add_bool_flag(parser, 'test', True)

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

    parser.add_argument('--lr', type=float)
    parser.add_argument('--noam_factor', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--l1_decay', type=float)
    parser.add_argument('--scheduler', choices=['attention'])
    parser.add_argument('--warm_up_steps', type=int)
    
    # --- LoRA core ---
    add_bool_flag(parser, 'train_lora', False)
    parser.add_argument("--lora_rank_qkv", type=int,
                        help="LoRA rank for Q/K/V projections in cross-attention (0 disables).")
    parser.add_argument("--lora_rank_out", type=int,
                        help="LoRA rank for attention output projection (0 disables).")
    parser.add_argument("--lora_rank_fc", type=int,
                        help="LoRA rank for final fc head (0 disables).")

    parser.add_argument("--lora_scale_qkv", type=float,
                        help="Scaling alpha for Q/K/V LoRA delta.")
    parser.add_argument("--lora_scale_out", type=float,
                        help="Scaling alpha for attention out LoRA delta.")
    parser.add_argument("--lora_scale_fc", type=float,
                        help="Scaling alpha for fc LoRA delta.")

    add_bool_flag(parser, 'lora_enable_attn', True)
    add_bool_flag(parser, 'lora_enable_fc', True)
    add_bool_flag(parser, 'lora_enable_self_attn', False)

    # --- Adapter training / mgmt ---
    parser.add_argument("--adapter_dir", type=str,
                        help="Directory to save/load LoRA-only adapter files.")
    parser.add_argument("--train_adapter_for_combo", nargs='+', choices=['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mw', 'formula'],
                        help="Canonical combo key to train an adapter for (e.g., 'c_nmr+hsqc+mw'). Empty = normal training.")

    add_bool_flag(parser, 'lora_only', True)
    parser.add_argument("--lora_lr", type=float,
                        help="Learning rate for LoRA params.")
    parser.add_argument("--lora_weight_decay", type=float,
                        help="Weight decay for LoRA params.")

    parser.add_argument("--full_mix_ratio", type=float,
                        help="Fraction of full-modality batches mixed in during adapter training (0..1).")
    parser.add_argument("--distill_full_alpha", type=float,
                        help="Weight of distillation loss on full-modality batches (protects full-data behavior).")
    parser.add_argument("--distill_target", type=str, choices=["logits", "embedding"],
                        help="Distill base behavior using logits or CLS embeddings on full-modality batches.")

    parser.add_argument("--lambda_hybrid", type=float)
    parser.add_argument("--fp_type", type=str, choices=['RankingEntropy', 'RankingSuperclass', 'RankingGlobal', 'RankingBalanced', 'Biosynfoni'])
    parser.add_argument("--arch", type=str, choices=['v1', 'v2'])

    args = parser.parse_args()
    if args.train_lora:
        assert args.train_adapter_for_combo is not None, 'Must select a combo to train the adapter on!'
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

    return SPECTREArgs(**args_dict)