import os
import json
from typing import Union, Annotated

import tyro

from .core.const import DO_NOT_OVERRIDE
from .marina import MARINAArgs
from .spectre import SPECTREArgs

ArchArgs = Union[
    Annotated[MARINAArgs, tyro.conf.subcommand("marina")],
    Annotated[SPECTREArgs, tyro.conf.subcommand("spectre")],
]


def parse_args() -> ArchArgs:
    """
    Parse command-line arguments using tyro.
    Supports subcommands: 'marina' or 'spectre'
    
    Usage:
        python main.py marina --batch_size 64 --lr 1e-4
        python main.py spectre --batch_size 32
    """
    args: ArchArgs = tyro.cli(
        ArchArgs,
        args=None,
    )
    
    # Handle checkpoint loading
    if getattr(args, "load_from_checkpoint", None):
        checkpoint_dir = os.path.dirname(args.load_from_checkpoint)
        params_path = os.path.join(checkpoint_dir, "params.json")
        if not os.path.exists(params_path):
            raise FileNotFoundError(
                f"No params.json found in checkpoint directory: {params_path}"
            )

        with open(params_path, "r") as f:
            checkpoint_args_dict = json.load(f)

        # Update args with checkpoint values, excluding protected fields
        for k, v in checkpoint_args_dict.items():
            if k in DO_NOT_OVERRIDE:
                continue
            if hasattr(args, k):
                setattr(args, k, v)
    
    # Set debug epochs if needed
    if getattr(args, "debug", False) and getattr(args, "epochs", None) is None:
        args.epochs = 1
    
    # Handle scheduler 'none' -> None conversion
    if hasattr(args, "scheduler") and args.scheduler == "none":
        args.scheduler = None
    
    return args
