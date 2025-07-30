from ..settings import Args
from .entropy import EntropyFPLoader

def get_fp_loader(args: Args):
    if args.fp_type == 'Entropy':
        fp_loader = EntropyFPLoader(args)
        fp_loader.setup(args.out_dim, args.fp_radius)
        return fp_loader
    elif args.fp_type == 'HYUN':
        return None
    else:
        raise NotImplementedError('MFP type not yet implemented')