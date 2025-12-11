import os
import json
import wandb

from ..log import get_logger, setup_file_logging, is_main_process

from ..core.const import WANDB_API_KEY_FILE
from ..marina.args import MARINAArgs
from ..spectre.args import SPECTREArgs

# Get logger after imports to avoid circular dependency
logger = get_logger(__file__)


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
        setup_file_logging(logger, os.path.join(results_path, "logs.txt"))
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
            setup_file_logging(logger, os.path.join(results_path, "logs.txt"))

        wandb_run = None

    return wandb_run
