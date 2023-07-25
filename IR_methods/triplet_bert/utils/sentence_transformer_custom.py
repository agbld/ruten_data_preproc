import os
import re
import json
from collections import OrderedDict
from typing import Dict, Tuple, Iterable, Type, Callable, List, Union
import transformers
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import trange

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.model_card_templates import ModelCardTemplate
from sentence_transformers.util import fullname

from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizerFast
# import config

# -------------------------------------------------------
#   Inherit SentenceTransformer to modify
# -------------------------------------------------------
class SentenceTransformerCustom(SentenceTransformer):
    # create tensorboard for monitoring

    def __init__(self, *args, tokenizer_pretrain_model_path: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        # writer = SummaryWriter(log_dir=writer_path)
        loss_step_log = 0
        metric_step_log = 0
        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_pretrain_model_path)

    # -------------------------------------------------------
    #   Modify fit function
    # -------------------------------------------------------
    def fit(self,
            train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
            evaluator: SentenceEvaluator = None,
            epochs: int = 1,
            steps_per_epoch = None,
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = transformers.AdamW,
            optimizer_params : Dict[str, object]= {'lr': 2e-5},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            show_progress_bar: bool = True,
            checkpoint_path: str = None,
            checkpoint_save_steps: int = 500,
            checkpoint_save_total_limit: int = 0
            ):
            
        ##Add info to model card
        #info_loss_functions = "\n".join(["- {} with {} training examples".format(str(loss), len(dataloader)) for dataloader, loss in train_objectives])
        info_loss_functions =  []
        for dataloader, loss in train_objectives:
            info_loss_functions.extend(ModelCardTemplate.get_train_objective_info(dataloader, loss))
        info_loss_functions = "\n\n".join([text for text in info_loss_functions])

        info_fit_parameters = json.dumps({"evaluator": fullname(evaluator), "epochs": epochs, "steps_per_epoch": steps_per_epoch, "scheduler": scheduler, "warmup_steps": warmup_steps, "optimizer_class": str(optimizer_class),  "optimizer_params": optimizer_params, "weight_decay": weight_decay, "evaluation_steps": evaluation_steps, "max_grad_norm": max_grad_norm }, indent=4, sort_keys=True)
        self._model_card_text = None
        self._model_card_vars['{TRAINING_SECTION}'] = ModelCardTemplate.__TRAINING_SECTION__.replace("{LOSS_FUNCTIONS}", info_loss_functions).replace("{FIT_PARAMETERS}", info_fit_parameters)


        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.to(self._target_device)

        dataloaders = [dataloader for dataloader, _ in train_objectives]

        for dataloader in dataloaders:
            dataloader.collate_fn = self.smart_batching_collate
        # -------------------------------------------------------
        #   Modify collate function if using ner augmentation
        # -------------------------------------------------------
        # for dataloader in dataloaders:
        #     # use ner collate function
        #     if config.ner_augment: dataloader.collate_fn = self.ner_collate_fn
        #     # use smart batching (original)
        #     else: dataloader.collate_fn = self.smart_batching_collate
        # -------------------------------------------------------

        loss_models = [loss for _, loss in train_objectives]
        for loss_model in loss_models:
            loss_model.to(self._target_device)

        self.best_score = -9999999

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        num_train_steps = int(steps_per_epoch * epochs)

        # Prepare optimizers
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
            scheduler_obj = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)

        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        num_train_objectives = len(train_objectives)

        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0

            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                for train_idx in range(num_train_objectives):
                    loss_model = loss_models[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)
                        
                    features, labels = data
                    # ##############################02/05 用於 ner 與 semantic 的特徵混合
                    # print("len(features): ", len(features))
                    # print("len(features)[0]: ", len(features[0]))
                    # print("features[0]: ", features[0]['input_ids'].shape)
                    # print("features[0]: ", features[0])
                    # # print("features[1]: ", features[1])
                    # # print("features[2]: ", features[2])
                    # input('wait')
                    # ##############################02/05

                    if use_amp:
                        with autocast():
                            loss_value = loss_model(features, labels, epoch)

                        scale_before_step = scaler.get_scale()
                        scaler.scale(loss_value).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()

                        skip_scheduler = scaler.get_scale() != scale_before_step
                    else:
                        loss_value = loss_model(features, labels, epoch)
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        optimizer.step()

                    optimizer.zero_grad()

                    if not skip_scheduler:
                        scheduler.step()
                
                # -------------------------------------------------------
                #   Output loss log on tensorboard
                # -------------------------------------------------------
                self.writer.add_scalar('Loss/Train', loss_value, self.loss_step_log)
                self.loss_step_log += 1
                # -------------------------------------------------------

                training_steps += 1
                global_step += 1

                if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps, callback, evaluation_steps)

                    for loss_model in loss_models:
                        loss_model.zero_grad()
                        loss_model.train()

                if checkpoint_path is not None and checkpoint_save_steps is not None and checkpoint_save_steps > 0 and global_step % checkpoint_save_steps == 0:
                    self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)

            self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback, evaluation_steps)

        if evaluator is None and output_path is not None:   #No evaluator, but output path: save final model version
            self.save(output_path)

        if checkpoint_path is not None:
            self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)
    
    # -------------------------------------------------------
    #   Modify eval during training
    # -------------------------------------------------------
    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback, evaluation_steps):
        eval_path = output_path
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            eval_path = os.path.join(output_path, "eval")
            os.makedirs(eval_path, exist_ok=True)

        if evaluator is not None:
            score = evaluator(self, output_path=eval_path, epoch=epoch, steps=steps)
            # -------------------------------------------------------
            #   Output score log on tensorboard
            # -------------------------------------------------------
            if steps != -1: 
                self.metric_step_log += evaluation_steps
                self.writer.add_scalar('Metric/3-levels-NDCG@50', score, self.metric_step_log)
            # -------------------------------------------------------
            if callback is not None:
                callback(score, epoch, steps)
            if score > self.best_score:
                self.best_score = score
                if save_best_model:
                    self.save(output_path)
            
