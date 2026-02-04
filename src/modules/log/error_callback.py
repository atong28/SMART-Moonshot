import traceback

import pytorch_lightning as pl

from .get_logger import get_logger


class ErrorLoggingCallback(pl.Callback):
    def on_exception(self, trainer, pl_module, err: BaseException) -> None:  # type: ignore[override]
        logger = get_logger(__file__)
        logger.error("[Main] Exception inside Lightning Trainer: %r", err)
        tb = "".join(traceback.format_exception(type(err), err, err.__traceback__))
        logger.error(tb)

