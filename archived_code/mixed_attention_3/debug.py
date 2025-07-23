from .src.settings import Args
from .src.dataset import MoonshotDataModule
from .src.model import SPECTRE, OptionalInputSPECTRE

def debug(args: Args, data_module: MoonshotDataModule, model: SPECTRE | OptionalInputSPECTRE):
    pass