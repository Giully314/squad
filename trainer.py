from dataclasses import dataclass

import torch

import log_utils

import config

from tqdm import tqdm 

import os
import utils

import wandb

@dataclass
class Trainer:
    proj_conf: config.ProjectConfig


    def train(self, model, opt, loss, train_dl, valid_dl):
        logger = log_utils.get_logger("squad")
        logger.info("Start training")
        paths = self.proj_conf.paths
        train = self.proj_conf.train
        epochs = train.epochs
        device = self.proj_conf.generic.device
        # lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(opt, lambda epoch: 1.1)
        sched_cfg = self.proj_conf.scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, sched_cfg.t_max, sched_cfg.min_lr)
        steps = 0

        best_valid_loss = 100.0

        if paths.checkpoint_dir is not None:
            paths.checkpoint_dir = os.path.join(paths.output_dir, paths.checkpoint_dir)
            utils.create_dir(paths.checkpoint_dir)

        for epoch in range(epochs):
            # train
            epoch_loss = 0
            model.train()   
            total_examples = 0
            with tqdm(train_dl, unit="batch") as progress_bar:
                for context, context_char, query, query_char, y1, y2, ids in progress_bar:    
                    progress_bar.set_description(f"Train epoch {epoch}")
                    batch_size = context.shape[0]
                    total_examples += batch_size
                    
                    # forward
                    context, context_char = context.to(device, non_blocking=True), context_char.to(device, non_blocking=True)
                    query, query_char = query.to(device, non_blocking=True), query_char.to(device, non_blocking=True)
                    y1, y2 = y1.to(device, non_blocking=True), y2.to(device, non_blocking=True)
                    y1_pred, y2_pred = model(context, context_char, query, query_char)
                    l = loss(y1_pred, y1, y2_pred, y2)
                    epoch_loss += l.item() * batch_size

                    # backward
                    l.backward()
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    # lr_scheduler.step()

                    # log info
                    steps += batch_size
                    progress_bar.set_postfix(loss=l.item())
                    wandb.log({"train_batch_loss": l.item()})
            # scheduler.step()
            wandb.log({"train_epoch_loss": epoch_loss/total_examples})

            # eval
            if train.evaluate_every_n_epochs is not None and epoch % train.evaluate_every_n_epochs == 0:
                model.eval()
                valid_loss = 0
                total_examples = 0
                with torch.no_grad():
                    with tqdm(valid_dl, unit="batch") as progress_bar:
                        for context, context_char, query, query_char, y1, y2, ids in progress_bar: 
                            progress_bar.set_description(f"Valid epoch {epoch}")
                            batch_size = context.shape[0]
                            total_examples += batch_size
                    
                            context, context_char = context.to(device, non_blocking=True), context_char.to(device, non_blocking=True)
                            query, query_char = query.to(device, non_blocking=True), query_char.to(device, non_blocking=True)
                            y1, y2 = y1.to(device, non_blocking=True), y2.to(device, non_blocking=True)

                            y1_pred, y2_pred = model(context, context_char, query, query_char)
                
                            l = loss(y1_pred, y1, y2_pred, y2)
                            valid_loss += l.item() * batch_size
                            
                            progress_bar.set_postfix(valid_loss=l.item())
                            wandb.log({"valid_batch_loss": l.item()})

                    valid_loss /= total_examples
                    wandb.log({"valid_epoch_loss": valid_loss})

                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        self.save_model(model, opt, paths.checkpoint_dir)
                

    def save_model(self, model, opt, dir):
        logger = log_utils.get_logger("squad")
        logger.info("Saving model")
        checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': opt.state_dict(),
                    'model_args': model.config,
                    'iter_num': self.proj_conf.train.epochs,
                    # 'best_val_loss': best_val_loss,
                    # 'config': config,
                }
        torch.save(checkpoint, os.path.join(dir, "model.pt"))


