import os


def is_main_process() -> int:
    return int(os.environ.get("RANK", 0)) == 0
