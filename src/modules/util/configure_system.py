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
        message=r"It is recommended to use `self\.log\(.+sync_dist=True\)` when logging on epoch level.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The '(train|val|test)_dataloader' does not have many workers which may be a bottleneck\..*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Checkpoint directory .* exists and is not empty\..*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Sparse CSR tensor support is in beta state\..*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=(
            r"The number of training batches \(\d+\) is smaller than the logging interval "
            r"Trainer\(log_every_n_steps=\d+\)\. Set a lower value for log_every_n_steps if you "
            r"want to see logs for the training epoch\."
        ),
        category=UserWarning,
        module="pytorch_lightning.loops.fit_loop",
    )
    warnings.filterwarnings(
        "ignore",
        message=(
            r"functools\.partial will be a method descriptor in future Python versions; "
            r"wrap it in enum\.member\(\) if you want to preserve the old behavior"
        ),
        category=FutureWarning,
        module="torch.distributed.algorithms.ddp_comm_hooks",
    )

    os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

    return
