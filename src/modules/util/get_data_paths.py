import os
from typing import Tuple

from ..core.const import DATASET_ROOT, PVC_ROOT
from ..marina.args import MARINAArgs
from ..spectre.args import SPECTREArgs


def get_data_paths(args: MARINAArgs | SPECTREArgs, today: str) -> Tuple[str, str]:
    """_summary_

    Args:
        args (MARINAArgs | SPECTREArgs): _description_
        today (str): _description_

    Returns:
        Tuple[str, str]: _description_
    """
    results_path = os.path.join(
        DATASET_ROOT,
        "results",
        args.experiment_name,
        today
    )

    final_path = os.path.join(
        PVC_ROOT,
        "results",
        args.experiment_name,
        today
    )

    return results_path, final_path
