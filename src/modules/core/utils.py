import torch
from functools import wraps

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
