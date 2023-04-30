from dataclasses import dataclass

import config

import torch

from tqdm import tqdm

import utils

import os
import csv
import log_utils

from collections import OrderedDict

import wandb

import numpy as np
import json


@dataclass
class Tester:
    proj_conf: config.ProjectConfig


    def test(self, model, loss, dl):
        logger = log_utils.get_logger("squad")
        data_table = wandb.Table(columns=["context", "question", "answer", "prediction"])
        device = self.proj_conf.generic.device
        model.to(device)
        model.eval()
 
        test_out_dir = os.path.join(self.proj_conf.paths.output_dir, self.proj_conf.paths.test_out_dir)
        utils.create_dir(test_out_dir)
        test_filename = os.path.join(self.proj_conf.paths.data_dir, self.proj_conf.test.test_file)
        test_eval = utils.load(test_filename)

        nll_meter = utils.AverageMeter()
        pred_dict = {}  # Predictions for TensorBoard
        sub_dict = {}   # Predictions for submission
        num_visuals = self.proj_conf.test.num_visuals

        with torch.no_grad():
            with tqdm(dl, unit="batch") as progress_bar:
                for context, context_char, query, query_char, y1, y2, ids in progress_bar: 
                    progress_bar.set_description(f"Test")
            
                    context, context_char = context.to(device, non_blocking=True), context_char.to(device, non_blocking=True)
                    query, query_char = query.to(device, non_blocking=True), query_char.to(device, non_blocking=True)
                    y1, y2 = y1.to(device, non_blocking=True), y2.to(device, non_blocking=True)

                    # the predictions are log_probs
                    y1_pred, y2_pred = model(context, context_char, query, query_char)
       
                    l = loss(y1_pred, y1, y2_pred, y2)
                    nll_meter.update(l.item(), context.shape[0])
        
                    # get probabilities
                    p1, p2 = y1_pred.exp(), y2_pred.exp()
                    starts, ends = utils.discretize(p1, p2, no_answer=True)

                    idx2pred, uuid2pred = utils.convert_tokens(test_eval,
                                                      ids.tolist(),
                                                      starts.tolist(),
                                                      ends.tolist(),
                                                      True)
                    pred_dict.update(idx2pred)
                    sub_dict.update(uuid2pred)
        
        results = utils.eval_dicts(test_eval, pred_dict, True)
        results_list = [('NLL', nll_meter.avg),
                        ('F1', results['F1']),
                        ('EM', results['EM'])]
       
        results_list.append(('AvNA', results['AvNA']))
        results = OrderedDict(results_list)
        wandb.log(results)

        # Log to console
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
        logger.info(f'{results_str}')

        # Log to wandb
        if num_visuals > len(pred_dict):
            num_visuals = len(pred_dict)
        visual_ids = np.random.choice(list(pred_dict), size=num_visuals, replace=False)

        logging_data = []
        with open(test_filename, 'r') as eval_file:
            eval_dict = json.load(eval_file)
        for i, id_ in enumerate(visual_ids):
            pred = pred_dict[id_] or 'N/A'
            example = eval_dict[str(id_)]
            question = example['question']
            context = example['context']
            answers = example['answers']

            gold = answers[0] if answers else 'N/A'    
            logging_data.append((context, question, gold, pred))
            data_table.add_data(context, question, gold, pred)

        with open("test.txt", "w") as f:
            for c, q, g, p in logging_data:
                tbl_fmt = (f'- **Question:** {q}\n'
                    + f'- **Context:** {c}\n'
                    + f'- **Answer:** {g}\n'
                    + f'- **Prediction:** {p}\n\n')
                f.write(tbl_fmt)
            



        wandb.log({"test_predictions": data_table})
        sub_path = os.path.join(test_out_dir, "result.csv")
        # log.info(f'Writing submission file to {sub_path}...')
        with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
            csv_writer = csv.writer(csv_fh, delimiter=',')
            csv_writer.writerow(['Id', 'Predicted'])
            for uuid in sorted(sub_dict):
                csv_writer.writerow([uuid, sub_dict[uuid]])