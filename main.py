import log_utils 

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

import config
import trainer
import tester
from model import BiDAFModel

import torch
import numpy as np
import random

import os
import utils

from datetime import datetime 

import wandb

cs = ConfigStore.instance()
cs.store("bidaf_config", node=config.ProjectConfig)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def my_app(cfg: config.ProjectConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

    # fix random seed 
    # TODO: fix randomness also for operations and dataloader?
    if cfg.generic.fix_random:
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

    if not cfg.generic.debug:
        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.emit_nvtx(False)

    train_conf = {
        "epochs": cfg.train.epochs,
        "batch_size": cfg.dataloader.batch_size,
        "learning_rate": cfg.optimizer.lr,
        "weight_decay": cfg.optimizer.weight_decay,
        "char_cnn_kernel_width": cfg.model.char_cnn_kernel_width,
        "char_cnn_channels": cfg.model.char_cnn_channels,
        "hidden_dim": cfg.model.hidden_dim,
        "contextual_layers": cfg.model.contextual_layers,
        "contextual_dropout": cfg.model.contextual_dropout ,
        "attention_dropout": cfg.model.attention_dropout,
        "modeling_layers": cfg.model.modeling_layers,
        "modeling_dropout": cfg.model.modeling_dropout,
    }
    
    wandb.init(
        project="bidaf",
        # config=os.path.join("configs", "config.yaml") # pass the yaml file
        config = train_conf,
        # mode="disabled"
    )

    # build outputdir based on the current trial day and hour 
    date = datetime.today().strftime('%Y-%m-%d %H:%M')
    output_dir = os.path.join(cfg.paths.output_dir, date)
    utils.create_dir(output_dir)
    cfg.paths.output_dir = output_dir

    # logging
    log_utils.create_logger("squad", output_dir)

    logger = log_utils.get_logger("squad")

    # load train/valid data 
    train_ds = utils.create_dataset(cfg, "train")
    train_dl = utils.create_dataloader(cfg, train_ds)
    
    valid_ds = None
    valid_dl = None
    if cfg.train.evaluate_every_n_epochs is not None:
        valid_ds = utils.create_dataset(cfg, "valid")
        valid_dl = utils.create_dataloader(cfg, valid_ds)
    

    # build the model, optimizer and loss and register the model to wandb 
    model = BiDAFModel(cfg)
    model.to(cfg.generic.device)
    wandb.watch(model, log="all", log_freq=len(train_dl)//4)

    optimizer = model.get_simple_optimizer(cfg)
    loss = utils.BiDAFLoss()
    
    # train
    train = trainer.Trainer(cfg)
    train.train(model, optimizer, loss, train_dl, valid_dl)

    if cfg.test.should_test:
        test = tester.Tester(cfg)
        test.test(model, loss, train_dl)

if __name__ == "__main__":
    my_app()
    wandb.finish()