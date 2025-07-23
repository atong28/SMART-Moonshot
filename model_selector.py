from importlib import import_module

def train(model_mode, *args, **kwargs):
    train_fn = getattr(import_module(f'{model_mode}_attention.train'), 'train')
    if train_fn is None:
        raise NotImplementedError()
    return train_fn(*args, **kwargs)

def test(model_mode, *args, **kwargs):
    test_fn = getattr(import_module(f'{model_mode}_attention.test'), 'test')
    if test_fn is None:
        raise NotImplementedError()
    return test_fn(*args, **kwargs)

def visualize(model_mode, *args, **kwargs):
    visualize_fn = getattr(import_module(f'{model_mode}_attention.visualize'), 'visualize')
    if visualize_fn is None:
        raise NotImplementedError()
    return visualize_fn(*args, **kwargs)

def get_fp_loader(model_mode, *args, **kwargs):
    fp_loader_fn = getattr(import_module(f'{model_mode}_attention.src.fp_loaders'), 'get_fp_loader')
    if fp_loader_fn is None:
        raise NotImplementedError()
    return fp_loader_fn(*args, **kwargs)

def build_model(model_mode, *args, **kwargs):
    model_fn = getattr(import_module(f'{model_mode}_attention.src.model'), 'build_model')
    if model_fn is None:
        raise NotImplementedError()
    return model_fn(*args, **kwargs)

def build_dataset(model_mode, *args, **kwargs):
    dataset_fn = getattr(import_module(f'{model_mode}_attention.src.dataset'), 'MoonshotDataModule')
    if dataset_fn is None:
        raise NotImplementedError()
    return dataset_fn(*args, **kwargs)
