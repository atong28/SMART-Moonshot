import torch
from functools import wraps
import logging
import sys

'''
Implementing L1 decay 
https://stackoverflow.com/questions/42704283/l1-l2-regularization-in-pytorch
'''
class L1(torch.nn.Module):
    def __init__(self, module, weight_decay):
        super().__init__()
        self.module = module
        self.weight_decay = weight_decay

        # Backward hook is registered on the specified module
        self.hook = self.module.register_full_backward_hook(self._weight_decay_hook)

    # Not dependent on backprop incoming values, placeholder
    def _weight_decay_hook(self, *_):
        for param in self.module.parameters():
            # If there is no gradient or it was zeroed out
            # Zeroed out using optimizer.zero_grad() usually
            # Turn on if needed with grad accumulation/more safer way
            # if param.grad is None or torch.all(param.grad == 0.0):

            # Apply regularization on it
            param.grad = self.regularize(param)

    def regularize(self, parameter):
        # L1 regularization formula
        return self.weight_decay * torch.sign(parameter.data)

    def forward(self, *args, **kwargs):
        # Simply forward and args and kwargs to module
        return self.module(*args, **kwargs)

'''
A wrapper to set float32 matmul precision to highest for all methods in a class.
Especially used for ranker class.
'''
def set_float32_highest_precision(cls):
    """Class decorator to set float32 matmul precision to highest for all methods."""
    for attr_name, attr_value in cls.__dict__.items():
        if callable(attr_value):
            setattr(cls, attr_name, wrap_method(attr_value))
    return cls

def wrap_method(method):
    @wraps(method)
    def wrapper(*args, **kwargs):
        # Set matmul precision to highest
        torch.set_float32_matmul_precision('highest')
        try:
            result = method(*args, **kwargs)
        finally:
            # Reset matmul precision to default after method execution
            torch.set_float32_matmul_precision('high')
        return result
    return wrapper

def get_debug_logger():
    logger = logging.getLogger("lightning")
    logger.setLevel(logging.INFO)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s: %(name)s: %(message)s')
    stdout_handler.setFormatter(formatter)

    if not any(isinstance(h, logging.StreamHandler) and h.stream == sys.stdout for h in logger.handlers):
        logger.addHandler(stdout_handler)

    return logger

def get_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    """
    Create a linear beta schedule:
      β₁ = beta_start, β_T = beta_end, with T total steps.
    Returns a 1-D tensor of shape [timesteps].
    """
    return torch.linspace(beta_start, beta_end, timesteps)

def q_sample(
    x0: torch.Tensor,
    t: torch.LongTensor,
    noise: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    batch_idx: torch.LongTensor
) -> torch.Tensor:
    """
    Sample x_t given x0 via the forward (q) process:
      x_t = sqrt(ᾱ_t) * x0 + sqrt(1 − ᾱ_t) * noise
    where ᾱ_t = alphas_cumprod[t].

    Args:
      x0            (N_nodes, D): clean node features
      t      (batch_size,):       integer timesteps for each graph
      noise   (N_nodes, D):       iid N(0,1) per node
      alphas_cumprod (T,):        precomputed cumulative product of (1−β)
      batch_idx (N_nodes,):       which graph each node belongs to (in [0..batch_size))

    Returns:
      x_t      (N_nodes, D): noisy features at step t
    """
    # gather ᾱ_t for each graph, then map to each node
    alpha_bar = alphas_cumprod[t]                     # → (B,)
    sqrt_ab   = alpha_bar.sqrt()[batch_idx]           # → (N_nodes,)
    sqrt_1m   = (1.0 - alpha_bar).sqrt()[batch_idx]    # → (N_nodes,)

    return x0 * sqrt_ab.unsqueeze(-1) + noise * sqrt_1m.unsqueeze(-1)