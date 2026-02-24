import os
from datetime import datetime

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
    configure_system,
    set_global_seed,
    get_data_paths,
    configure_wandb,
    write_results
)
from .modules.log import get_logger

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


def main():
    configure_system()
    # create timestamp for run (allow overriding so launchers can pre-create
    # the results directory and capture stdout/stderr into it)
    today = os.environ.get("SMART_RUN_ID") or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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

    logger = get_logger(__file__)

    # create a wandb run
    wandb_run = configure_wandb(args, results_path, today)

    # train a model using the args as input
    if args.train:
        train(
            args,
            data_module,
            model,
            results_path,
            wandb_run=wandb_run,
            fp_loader=fp_loader
        )
    elif args.test:
        test(
            args,
            data_module,
            model,
            results_path,
            ckpt_path=args.load_from_checkpoint,
            wandb_run=wandb_run,
            fp_loader=fp_loader
        )
    elif args.benchmark:
        benchmark(
            args,
            data_module,
            model,
            fp_loader,
            wandb_run=wandb_run,
            load_from_checkpoint=args.load_from_checkpoint
        )
    else:
        raise ValueError("[Main] Nothing to do!")

    # if it was a training run write out the results to a path and end the wandb run
    write_results(args, final_path, results_path, logger, wandb_run)


if __name__ == "__main__":
    main()
