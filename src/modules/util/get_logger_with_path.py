import os
import sys
import logging

from .is_main_process import is_main_process

logger = logging.getLogger("lightning")
logger.setLevel(logging.INFO if is_main_process() else logging.WARNING)


def get_logger_with_path(path: str) -> logging.Logger:
    """_summary_

    Args:
        path (str): _description_

    Returns:
        Logger: _description_
    """
    if not logger.handlers:
        os.makedirs(path, exist_ok=True)

        file_path = os.path.join(path, "logs.txt")

        with open(file_path, "w"):
            pass

        fmt = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh = logging.FileHandler(file_path)
        fh.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(logging.StreamHandler(sys.stdout))

    return logger
