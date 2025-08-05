import os
import json
from datetime import datetime
import torch
import logging
import sys
import pickle

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from diffms.diffusion.extra_features import ExtraFeatures
from diffms.metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
from diffms.diffusion.extra_features_molecular import ExtraMolecularFeatures
from diffms.analysis.visualization import MolecularVisualization

from settings import SpectreArgs, MoonshotArgs
from spectre.model import SPECTRE
from dataset import MoonshotDataModule
from dataset_infos import MoonshotInfos
from model import Moonshot

torch.set_float32_matmul_precision('medium')

def is_main_process():
    return int(os.environ.get("RANK", 0)) == 0

def init_logger(path):
    logger = logging.getLogger("lightning")
    if is_main_process():
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
    if not logger.handlers:
        file_path = os.path.join(path, "logs.txt")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as fp:
            pass

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(file_path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger

if __name__ == '__main__':
    use_cache = True
    dump_cache = False
    if use_cache and os.path.exists('cache.pkl'):
        moonshot_args, spectre_path, spectre_args, datamodule, dataset_infos = pickle.load(open('cache.pkl', 'rb'))
        today = datetime.now().strftime("%Y-%m-%d")
        results_path = os.path.join(
            spectre_args.data_root, "results", moonshot_args.name, today
        )
        final_path = os.path.join(
            spectre_args.code_root, "results", moonshot_args.name, today
        )
        logger = init_logger(results_path)
        logger.info(f'[Main] Config: {moonshot_args}')
    else:
        moonshot_args = MoonshotArgs()
        spectre_path = moonshot_args.spectre_ckpt
        spectre_args = SpectreArgs(**json.load(open(os.path.join(spectre_path, 'params.json'), 'r')))
        today = datetime.now().strftime("%Y-%m-%d")
        results_path = os.path.join(
            spectre_args.data_root, "results", moonshot_args.name, today
        )
        logger = init_logger(results_path)
        logger.info(f'[Main] Config: {moonshot_args}')
        final_path = os.path.join(
            spectre_args.code_root, "results", moonshot_args.name, today
        )
        datamodule = MoonshotDataModule(spectre_args, moonshot_args, results_path)
        dataset_infos = MoonshotInfos(datamodule)
    if dump_cache:
        cache = [moonshot_args, spectre_path, spectre_args, datamodule, dataset_infos]
        pickle.dump(cache, open('cache.pkl', 'wb'))
    logger.info('[Main] Infos computed')
    spectre = SPECTRE(spectre_args)
    spectre.load_state_dict(torch.load(os.path.join(spectre_path, 'best.ckpt'), map_location='cpu', weights_only=True)['state_dict'])
    logger.info('[Main] SPECTRE backbone loaded')
    extra_features = ExtraFeatures(moonshot_args.extra_features, dataset_info=dataset_infos)
    domain_features = ExtraMolecularFeatures(dataset_infos)
    
    dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features, domain_features=domain_features)
    
    train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
    visualization_tools = MolecularVisualization(None, dataset_infos)
    moonshot = Moonshot(moonshot_args, dataset_infos, train_metrics, visualization_tools, extra_features, domain_features, spectre)
    logger.info('[Main] Moonshot loaded')
    callbacks = [LearningRateMonitor(logging_interval='step')]
    if moonshot_args.save_model:
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"checkpoints/{moonshot_args.name}", # best (top-5) checkpoints
            filename='{epoch}',
            monitor='val/NLL',
            save_top_k=1,
            mode='min',
            every_n_epochs=1)
        last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{moonshot_args.name}", filename='last', every_n_epochs=1) # most recent checkpoint
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback)
    name = moonshot_args.name
    if name == 'debug':
        logging.warning("Run is called 'debug' -- it will run with fast_dev_run. ")
    
    loggers = [
        CSVLogger(save_dir=f"logs/{name}", name=name),
        WandbLogger(name=f'{name}_{today}', save_dir=f"logs/{name}", project=moonshot_args.wandb_name, log_model=False, config=moonshot_args.__dict__)
    ]
    use_gpu = moonshot_args.gpus > 0 and torch.cuda.is_available()
    trainer = Trainer(
        gradient_clip_val=moonshot_args.clip_grad,
        strategy=DDPStrategy(find_unused_parameters=True), #, static_graph=True),
        accelerator='gpu' if use_gpu else 'cpu',
        devices=moonshot_args.gpus if use_gpu else 1,
        max_epochs=moonshot_args.n_epochs,
        check_val_every_n_epoch=moonshot_args.check_val_every_n_epochs,
        fast_dev_run=moonshot_args.name == 'debug',
        callbacks=callbacks,
        log_every_n_steps=50 if name != 'debug' else 1,
        logger=loggers
    )
    if not moonshot_args.test_only:
        trainer.fit(moonshot, datamodule=datamodule, ckpt_path=moonshot_args.resume)
        if moonshot_args.name not in ['debug', 'test']:
            trainer.test(moonshot, datamodule=datamodule)
    else:
        # Start by evaluating test_only_path
        trainer.test(moonshot, datamodule=datamodule, ckpt_path=moonshot_args.test_only)
