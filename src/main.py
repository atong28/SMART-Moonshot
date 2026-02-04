import os
from datetime import datetime
import torch

from .modules import (
    MARINA,
    MARINAArgs,
    MARINADataModule,
    SPECTRE,
    SPECTREArgs,
    SPECTREDataModule,
    parse_args,
    train,
    test,
    benchmark
)
from .modules.util import (
    set_global_seed,
    get_data_paths,
    configure_wandb,
    write_results
)
from .modules.log import get_logger, setup_file_logging

from .modules.data.fp_loader import make_fp_loader
from .modules.core.const import DATASET_ROOT


ARCH_MODEL_CLASSES = {
    "MARINA": MARINA,
    "SPECTRE": SPECTRE,
}

ARCH_DATAMODULE_CLASSES = {
    "MARINA": MARINADataModule,
    "SPECTRE": SPECTREDataModule,
}


def _log_and_reraise(logger, message: str) -> None:
    """
    Helper to log the current exception with a message and re-raise it.

    This should only be called from within an `except` block so that the
    original traceback is preserved.
    """
    logger.exception(message)
    # Re-raise the currently handled exception with its original traceback.
    raise


def main():
    # create timestamp for run
    today = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # parse the args
    args: MARINAArgs | SPECTREArgs = parse_args()
    # seed the libs
    set_global_seed(args.seed)
    #
    fp_loader = make_fp_loader(
        args.fp_type,
        entropy_out_dim=args.out_dim,
        retrieval_path=os.path.join(
            DATASET_ROOT,
            'retrieval.pkl'
        )
    )
    # create a mapping
    model_class = ARCH_MODEL_CLASSES[args.project_name]
    data_module_class = ARCH_DATAMODULE_CLASSES[args.project_name]
    # define the model and data module
    model: MARINA | SPECTRE = model_class(args, fp_loader)
    data_module: MARINADataModule | SPECTREDataModule = data_module_class(
        args,
        fp_loader
    )
    # paths to output data
    results_path, final_path = get_data_paths(args, today)

    # logger and file logging must be initialized before any work so that
    # uncaught exceptions always have a destination.
    logger = get_logger(__file__)
    setup_file_logging(logger, os.path.join(final_path, "logs.txt"))

    try:
        # create a wandb run
        wandb_run = configure_wandb(args, results_path, today)

        # train a model using the args as input
        if args.train:
            train(
                args,
                data_module,
                model,
                results_path,
                wandb_run=wandb_run
            )
        # test a given run against a set
        elif args.test:
            test(
                args,
                data_module,
                model,
                results_path,
                ckpt_path=args.load_from_checkpoint,
                wandb_run=wandb_run,
                sweep=True
            )
        elif args.benchmark:
            benchmark(
                args,
                data_module,
                model,
                fp_loader
            )
        else:
            raise ValueError(
                "[Main] Both --no_train and --no_test set; nothing to do!")

        # if it was a training run write out the results to a path and end the wandb run
        write_results(args, final_path, results_path, logger, wandb_run)
    except Exception as e:
        # torchrun aggregates worker failures into ChildFailedError on the
        # parent process; handle that case with a clearer message, but still
        # re-raise so job infrastructure sees a failure.
        if e.__class__.__name__ == "ChildFailedError":
            _log_and_reraise(logger, "[Main] ChildFailedError in distributed run; check worker logs for details.")
        else:
            _log_and_reraise(logger, "[Main] Unhandled exception in main()")


if __name__ == "__main__":
    main()
