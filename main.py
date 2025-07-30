import logging
import os
import sys
import json
import shutil
from datetime import datetime
import wandb

import numpy as np
import random
import torch
import pytorch_lightning as pl

from model_selector import (
    build_dataset,
    build_model,
    get_fp_loader,
    train,
    test,
    visualize,
    parse_args,
)

def is_main_process():
    return int(os.environ.get("RANK", 0)) == 0

def init_logger(path):
    logger = logging.getLogger("lightning")
    logger.setLevel(logging.INFO if is_main_process() else logging.WARNING)
    if not logger.handlers:
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, "logs.txt")
        with open(file_path, "w"):
            pass
        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh = logging.FileHandler(file_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger

def seed_everything(seed):
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    args, model_mode = parse_args()
    seed_everything(args.seed)

    # build a common results path
    today = datetime.now().strftime("%Y-%m-%d")
    results_path = os.path.join(
        args.data_root, "results", args.experiment_name, today
    )
    final_path = os.path.join(
        args.code_root, "results", args.experiment_name, today
    )
    experiment_id = f"{args.experiment_name}_{today}"

    # rank 0: create outputs, logging, wandb
    if is_main_process() and args.train and not args.visualize:
        os.makedirs(results_path, exist_ok=True)
        logger = init_logger(results_path)
        logger.info("[main] parsed args:\n%s", args)

        # dump params.json
        with open(os.path.join(results_path, "params.json"), "w") as fp:
            json.dump(vars(args), fp, indent=2)

        # login to wandb
        with open("wandb_api_key.json") as kf:
            key = json.load(kf)["key"]
        wandb.login(key=key)
        wandb_run = wandb.init(
            project="SPECTRE",
            name=experiment_id,
            config=vars(args),
            resume="allow",
        )
    else:
        wandb_run = None

    # instantiate data + model
    if model_mode == "moonshot_e2e":
        # diffusion end‐to‐end: no fingerprint loader
        data_module = build_dataset(model_mode, args, results_path)
        model = build_model(model_mode, args)
    else:
        # fixed‐input spectral ranking: needs an FP loader
        fp_loader   = get_fp_loader(model_mode, args)
        data_module = build_dataset(
            model_mode, args, results_path, fp_loader
        )
        model       = build_model(
            model_mode,
            args,
            optional_inputs=(set(args.requires) != set(args.input_types)),
            fp_loader=fp_loader,
            combinations_names=data_module.combinations_names,
        )

        # visualization shortcut
        if args.visualize and is_main_process():
            logger.info("[Main] Entering visualization mode")
            visualize(
                model_mode,
                data_module,
                model,
                ckpt_path=args.load_from_checkpoint,
            )
            sys.exit(0)

    # training or testing
    if args.train:
        train(model_mode, args, data_module, model, results_path, wandb_run=wandb_run)
    elif args.test:
        test(model_mode, args, data_module, model, results_path, ckpt_path=args.load_from_checkpoint, wandb_run=wandb_run)
    else:
        raise ValueError("[Main] Both --no_train and --no_test set; nothing to do!")

    # rank0: copy out and finish wandb
    if is_main_process():
        logger.info("[Main] Moving results to final destination")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        shutil.move(results_path, final_path)
        wandb.finish()
