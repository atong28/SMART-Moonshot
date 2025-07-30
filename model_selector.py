from importlib import import_module
import argparse

def train(model_mode, *args, **kwargs):
    train_fn = getattr(import_module(f'{model_mode}.train'), 'train')
    if train_fn is None:
        raise NotImplementedError()
    return train_fn(*args, **kwargs)

def test(model_mode, *args, **kwargs):
    test_fn = getattr(import_module(f'{model_mode}.test'), 'test')
    if test_fn is None:
        raise NotImplementedError()
    return test_fn(*args, **kwargs)

def visualize(model_mode, *args, **kwargs):
    visualize_fn = getattr(import_module(f'{model_mode}.visualize'), 'visualize')
    if visualize_fn is None:
        raise NotImplementedError()
    return visualize_fn(*args, **kwargs)

def get_fp_loader(model_mode, *args, **kwargs):
    fp_loader_fn = getattr(import_module(f'{model_mode}.src.fp_loaders'), 'get_fp_loader')
    if fp_loader_fn is None:
        raise NotImplementedError()
    return fp_loader_fn(*args, **kwargs)

def build_model(model_mode, *args, **kwargs):
    model_fn = getattr(import_module(f'{model_mode}.src.model'), 'build_model')
    if model_fn is None:
        raise NotImplementedError()
    return model_fn(*args, **kwargs)

def build_dataset(model_mode, *args, **kwargs):
    dataset_fn = getattr(import_module(f'{model_mode}.src.dataset'), 'MoonshotDataModule')
    if dataset_fn is None:
        raise NotImplementedError()
    return dataset_fn(*args, **kwargs)

def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--model_mode', required=True,
                            help='Which model mode to run')
    pre_args, remaining_argv = pre_parser.parse_known_args()
    mode = pre_args.model_mode
    parse_fn = getattr(import_module(f'{mode}.src.args'), 'parse_args')
    if parse_fn is None:
        raise NotImplementedError()
    return parse_fn(remaining_argv), mode