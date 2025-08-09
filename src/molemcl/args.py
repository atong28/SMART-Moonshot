# molemcl/args.py

import argparse
from .settings import MoleMCLArgs

def add_bool_flag(parser: argparse.ArgumentParser, name: str, default: bool):
    if default:
        parser.add_argument(f'--no_{name}', dest=name, action='store_false')
    else:
        parser.add_argument(f'--{name}', dest=name, action='store_true')
    parser.set_defaults(**{name: default})

def parse_args() -> MoleMCLArgs:
    p = argparse.ArgumentParser("MoleMCL Stage 2")

    # run id
    p.add_argument("--experiment_name", type=str)
    p.add_argument("--project_name", type=str)
    p.add_argument("--seed", type=int)
    add_bool_flag(p, "train", True)
    add_bool_flag(p, "test", False)
    p.add_argument("--load_from_checkpoint", type=str)

    # data
    p.add_argument("--ae_root", type=str)
    add_bool_flag(p, "train_on_duplicates", False)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--num_workers", type=int)
    add_bool_flag(p, "persistent_workers", True)
    add_bool_flag(p, "pin_memory", True)
    p.add_argument("--mask_rate", type=float)
    add_bool_flag(p, "mask_edge", True)

    # model
    p.add_argument("--num_layer", type=int)
    p.add_argument("--emb_dim", type=int)
    p.add_argument("--gnn_type", type=str, choices=["gin","gcn","gat","graphsage"])
    p.add_argument("--JK", type=str, choices=["last","sum","max","concat"])
    p.add_argument("--drop_ratio", type=float)
    p.add_argument("--alpha", type=float)
    p.add_argument("--temperature", type=float)

    # optim
    p.add_argument("--lr", type=float)
    p.add_argument("--weight_decay", type=float)

    # trainer
    p.add_argument("--max_epochs", type=int)
    p.add_argument("--accelerator", type=str, choices=["auto","cpu","gpu"])
    p.add_argument("--devices", type=int)
    p.add_argument("--log_every_n_steps", type=int)
    add_bool_flag(p, "deterministic", False)

    # I/O
    p.add_argument("--ckpt_path", type=str)
    p.add_argument("--output_dir", type=str)

    args = p.parse_args()
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    return MoleMCLArgs(**args_dict)
