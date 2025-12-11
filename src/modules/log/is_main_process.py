"""
Check if current process is the main process (rank 0).
Moved here to avoid circular imports - this is a low-level utility.
"""
import os

import torch.distributed as dist


def is_main_process() -> bool:
    """
    Check if current process is the main process (rank 0).
    
    Handles multiple distributed scenarios:
    - torchrun (RANK env var)
    - PyTorch Lightning (LOCAL_RANK env var)
    - torch.distributed (get_rank())
    """
    # Check environment variables first (torchrun, Lightning)
    rank = os.environ.get("RANK")
    if rank is not None:
        return int(rank) == 0
    
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        return int(local_rank) == 0
    
    # Check torch.distributed if initialized
    if dist.is_initialized():
        return dist.get_rank() == 0
    
    # Default: assume main process if no distributed setup
    return True

