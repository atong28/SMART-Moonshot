import os
import wandb
import shutil
import logging

from ..log import is_main_process
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

        # If a launcher pre-created final_path (e.g., to tee stdout/stderr there),
        # avoid nesting results under final_path/today by merging contents instead.
        if os.path.exists(final_path):
            if not os.path.isdir(final_path):
                raise RuntimeError(f"[Main] final_path exists but is not a directory: {final_path}")

            os.makedirs(final_path, exist_ok=True)
            for name in os.listdir(results_path):
                src = os.path.join(results_path, name)
                dst = os.path.join(final_path, name)
                if os.path.exists(dst):
                    shutil.rmtree(dst) if os.path.isdir(dst) else os.remove(dst)
                shutil.move(src, dst)
            os.rmdir(results_path)
        else:
            shutil.move(results_path, final_path)

        if wandb_run is not None:
            wandb.finish()

    return
