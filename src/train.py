import wandb
import json
from torch.utils.data import DataLoader, SubsetRandomSampler
from omegaconf import OmegaConf
# from dataloader.ArcFDI import DatasetPreload, collate_fn
# from model.ArcFDI import Net
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import os
import random
import argparse
import torch
# from trainer import Trainer
import pickle
import setproctitle
import logging
import json
import numpy as np

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch import Trainer

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO, 
                    datefmt='%Y-%m-%d %H:%M:%S')

num_cpus = os.cpu_count()
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.set_num_threads(1)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('medium')

parser = argparse.ArgumentParser()
parser.add_argument('--config_name',   '-cn', default='dev',  type=str)
parser.add_argument('--project_name',  '-pn', default='dev',  type=str)
parser.add_argument('--group_name',    '-gn', default='dev',  type=str)
parser.add_argument('--session_name',  '-sn', default='dev',  type=str)
parser.add_argument('--fast_dev_run',  '-fd', default=False,  action='store_true')

# Override Arguments
parser.add_argument('--random_seed',   '-rs', default=None,   type=int)
parser.add_argument('--dataset_split', '-ds', default=None,   type=str)

args = parser.parse_args()

def override_arguments(conf, args):
    if args.random_seed:
        conf.experiment.random_seed = args.random_seed
    if args.dataset_split:
        conf.dataprep.split = args.dataset_split

    return conf

def setup_seed(seed):
    logging.info(f"Setting the Random Seed to {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_wandb(conf):
    logging.info("Setting the Wandb Run Object...")
    wandb_init = dict()
    wandb_init['config']    = argparse.Namespace(**conf)
    wandb_init['project']   = args.project_name
    wandb_init['group']     = args.group_name
    wandb_init['name']      = args.session_name
    wandb_init['notes']     = args.session_name
    wandb_init['save_dir']  = f'{conf.path.checkpoint}/{args.project_name}-{args.group_name}-{args.session_name}'
    wandb_init['log_model'] = True
    os.environ['WANDB_START_METHOD'] = 'thread'

    return wandb_init

def reset_wandb_env():
    exclude = {'WANDB_PROJECT', 'WANDB_ENTITY', 'WANDB_API_KEY',}
    for k, v in os.environ.items():
        if k.startswith('WANDB_') and k not in exclude:
            del os.environ[k]

def load_dataset_collate_model(conf):
    if conf.model_params.model_type == 'arcdfi':
        from dataloader.ArcFDI import DatasetPreload, collate_fn
        from model.arcfdi.ArcDFI import Model
    else:
        raise ValueError("Invalid Model Type")
    
    dataset_train = DatasetPreload(conf, 'train')
    dataset_valid = DatasetPreload(conf, 'valid')
    dataset_test  = DatasetPreload(conf, 'test')
    datasets      = (dataset_train, dataset_valid, dataset_test)

    with torch.device('cpu'):
        model         = Model(conf)

    return datasets, collate_fn, model

def run(trainer, model, trainloader, validloader, testloader):

    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=validloader)
    trainer.test(model, dataloaders=testloader)

if __name__ == '__main__':
    logging.info(f"WANDB Project Name:       {args.project_name}")
    logging.info(f"WANDB Group Name:         {args.group_name}")
    logging.info(f"WANDB Session Name:       {args.session_name}")
    logging.info(f"OMEGA Configuration Name: {args.config_name}")
    conf = OmegaConf.load(f'./src/settings.yaml')[args.config_name]
    conf = override_arguments(conf, args)
    logging.info(json.dumps(OmegaConf.to_container(conf), indent=4))

    setup_seed(conf.experiment.random_seed) #  911012 940107 940404 961104 980227  991214 990220
    wandb_params = setup_wandb(conf)
    wandb_logger = WandbLogger(**wandb_params)

    datasets, collate_fn, model = load_dataset_collate_model(conf)

    early_stopping = EarlyStopping(
                monitor=conf.train_params.early_stopping.monitor,
                patience=conf.train_params.early_stopping.patience,
                verbose=True,
                mode=conf.train_params.early_stopping.mode
    )

    chpt_callback  = ModelCheckpoint(
                dirpath= wandb_params['save_dir'],
                monitor=conf.train_params.early_stopping.monitor,
                mode='min'
    )

    trainer        = Trainer(
                default_root_dir=conf.path.root,
                fast_dev_run=args.fast_dev_run,
                logger=wandb_logger, 
                num_sanity_val_steps=2,
                devices=torch.cuda.device_count(),
                enable_model_summary=True,
                inference_mode=True,
                accelerator=conf.train_params.accelerator,
                strategy=conf.train_params.strategy,
                enable_progress_bar=True,
                log_every_n_steps=5,
                gradient_clip_val=0.5,
                check_val_every_n_epoch=1,
                deterministic=True,
                enable_checkpointing=True,
                accumulate_grad_batches=conf.train_params.accumulate_grad_batches,
                max_epochs=conf.train_params.max_epochs,
                callbacks=[early_stopping, chpt_callback]
    )

    logging.info(len(datasets[0]))
    logging.info(len(datasets[1]))
    logging.info(len(datasets[2]))
    
    batch_size         = conf.train_params.batch_size
    batch_per_rank     = batch_size // torch.cuda.device_count()
    num_batch_per_rank = (len(datasets[0]) // batch_size) // torch.cuda.device_count()

    logging.info(f"Expected # of Batches to be Distributed in Each Rank: {num_batch_per_rank}")
    
    trainloader   = DataLoader(datasets[0], batch_size=batch_per_rank, collate_fn=collate_fn, num_workers=4, shuffle=True)
    validloader   = DataLoader(datasets[1], batch_size=batch_per_rank, collate_fn=collate_fn, num_workers=4)
    testloader    = DataLoader(datasets[2], batch_size=batch_per_rank, collate_fn=collate_fn, num_workers=4)

    run(trainer, model, trainloader, validloader, testloader)