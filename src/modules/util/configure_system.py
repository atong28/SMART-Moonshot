import os
import warnings


def configure_system():
    """
    Configure the runtime environment for deterministic and quiet execution.

    This function:
      - Forces PyTorch and other OpenMP-based libraries to use a single thread
        by setting ``OMP_NUM_THREADS=1``. This helps ensure reproducibility and
        avoids unexpected multi-threading overhead.
      - Suppresses specific PyTorch and PyTorch Lightning prototype or logging
        warnings that are known to be noisy but harmless.
      - Sets ``TORCH_CPP_LOG_LEVEL`` to ``ERROR`` to reduce low-level C++ backend
        logging from PyTorch.

    Returns:
        None
    """
    os.environ["OMP_NUM_THREADS"] = "1"

    warnings.filterwarnings(
        "ignore",
        message="The PyTorch API of nested tensors is in prototype stage",
        category=UserWarning,
        module="torch.nn.modules.transformer"
    )
    warnings.filterwarnings(
        "ignore",
        message=r"It is recommended to use `self\.log\('.*', \.\.\., sync_dist=True\)` when logging on epoch level",
        category=UserWarning,
        module="pytorch_lightning.trainer.connectors.logger_connector.result"
    )

    os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

    return
