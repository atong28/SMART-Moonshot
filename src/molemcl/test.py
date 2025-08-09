import pytorch_lightning as pl
from .settings import MoleMCLArgs
from ..dataset.molemcl import AEDataModule
from .model import MoleMCLModule

def test(args: MoleMCLArgs):
    dm = AEDataModule(
        ae_root=args.ae_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory,
        mask_rate=args.mask_rate,
        mask_edge=args.mask_edge,
        train_on_duplicates=args.train_on_duplicates,
    )

    model = MoleMCLModule.load_from_checkpoint(args.ckpt_path)
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        log_every_n_steps=args.log_every_n_steps,
        deterministic=args.deterministic,
    )
    trainer.test(model, datamodule=dm)
