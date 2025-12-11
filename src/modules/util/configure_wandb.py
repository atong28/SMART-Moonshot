import os
import json
import wandb

from .get_logger_with_path import get_logger_with_path
from .is_main_process import is_main_process

from ..core.const import WANDB_API_KEY_FILE
from ..marina.args import MARINAArgs
from ..spectre.args import SPECTREArgs


def configure_wandb(args: MARINAArgs | SPECTREArgs, results_path: str, today: str):
    """_summary_

    Args:
        args (MARINAArgs | SPECTREArgs): _description_
        results_path (str): _description_
        today (str): _description_

    Raises:
        RuntimeError: _description_

    Returns:
        _type_: _description_
    """
    experiment_id = f"{args.experiment_name}_{today}"

    if is_main_process() and args.train:
        os.makedirs(results_path, exist_ok=True)
        logger = get_logger_with_path(results_path)
        logger.info("[Main] Parsed args:\n%s", args)

        with open(os.path.join(results_path, "params.json"), "w") as fp:
            json.dump(vars(args), fp, indent=2)

        if not os.path.exists(WANDB_API_KEY_FILE):
            raise RuntimeError(
                f"WANDB API key file not found at {WANDB_API_KEY_FILE}")

        with open(WANDB_API_KEY_FILE) as kf:
            key = json.load(kf)["key"]

        wandb.login(key=key)

        wandb_run = wandb.init(
            project=args.project_name,
            name=experiment_id,
            config=vars(args),
            resume="allow",
        )
    else:
        # ensure path exists before creating a logger
        if is_main_process():
            os.makedirs(results_path, exist_ok=True)
            logger = get_logger_with_path(results_path)
        else:
            logger = None

        wandb_run = None

    return wandb_run
