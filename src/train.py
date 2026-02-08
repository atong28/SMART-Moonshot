import os
import wandb
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger


from .modules.err_correction.model import HSQC2SelfiesModel, LitHSQC2Selfies
from .modules.err_correction.dataset import HSQCSelfiesDataset
from .modules.err_correction.collate import collate_hsqc_selfies
from .modules.err_correction.vocab import SelfiesVocab

DATA_PATH = os.environ.get("DATA_DIR", "/data")
VOCAB_PATH = os.path.join(DATA_PATH, "selfies_vocab.json")

vocab = SelfiesVocab.from_json(VOCAB_PATH)
vocab_size = len(vocab)
pad_id: int = vocab.get_pad_id()
eos_id: int = vocab.get_eos_id()
bos_id: int = vocab.get_bos_id()
unk_id: int = vocab.get_unk_id()

model = HSQC2SelfiesModel(
        vocab_size=vocab_size,
        pad_id=pad_id,
        d_model=256,
        nhead=8,
        num_layers=4,
        dropout=0.1,
        max_len=512
    )

lit = LitHSQC2Selfies(model, pad_id=pad_id, bos_id=bos_id, eos_id=eos_id, lr=3e-4)

train_ds = HSQCSelfiesDataset(DATA_PATH, split="train", normalize_intensity=True)
val_ds   = HSQCSelfiesDataset(DATA_PATH, split="val", normalize_intensity=True)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=8,
                          collate_fn=lambda b: collate_hsqc_selfies(b, pad_id=pad_id), persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=8,
                        collate_fn=lambda b: collate_hsqc_selfies(b, pad_id=pad_id), persistent_workers=True)

wandb_logger = WandbLogger(project="smart-hsqc2selfies", name="metrics-update-hsqc2selfies")

trainer = pl.Trainer(
    max_epochs=10,
    accelerator="gpu",
    devices=1,
    precision="16-mixed",
    logger=wandb_logger,
    log_every_n_steps=50,
    gradient_clip_val=1.0,
)

trainer.fit(lit, train_loader, val_loader)
wandb.finish()