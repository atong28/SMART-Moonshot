import os
import wandb
import shutil
import logging

from .is_main_process import is_main_process
from ..marina.args import MARINAArgs
from ..spectre.args import SPECTREArgs


def write_results(
    args: MARINAArgs | SPECTREArgs,
    final_path: str,
    results_path: str,
    logger: logging.Logger = None,
    wandb_run=None
) -> None:
    """_summary_

    Args:
        args (MARINAArgs | SPECTREArgs): _description_
        final_path (str): _description_
        result_path (str): _description_
        logger (logging.Logger, optional): _description_. Defaults to None.
        wandb_run (_type_, optional): _description_. Defaults to None.
    """
    if is_main_process() and args.train:
        logger and logger.info("[Main] Moving results to final destination")

        os.makedirs(os.path.dirname(final_path), exist_ok=True)

        shutil.move(results_path, final_path)

        if wandb_run is not None:
            wandb.finish()

    return
