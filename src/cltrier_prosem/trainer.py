import logging
from datetime import datetime
from typing import Tuple, Dict, Callable

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

from .dataset import Dataset
from .metric import Metric


class Trainer:

    def __init__(
            self,
            dataset: Dict[str, Dataset],
            model: torch.nn.Module,
            collation_fn: Callable,
            config: dict
    ):
        self.dataset = dataset
        self.model = model
        self.collation_fn = collation_fn
        self.config: dict = config

        # progress tracking
        self.progress: dict = {
            'epoch': [],
            'loss_train': [],
            'loss_eval': [],
            'f1_train': [],
            'f1_eval': [],
            'duration': []
        }

        #  load optimizer and scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=self.config['learning_rate'])
        self.scheduler = get_scheduler(
            "linear", optimizer=self.optimizer, num_warmup_steps=0,
            num_training_steps=self.config['num_epochs'] * len(self.dataset['train']),
        )

    def __call__(self) -> None:
        best_eval_metric: Dict[str, pd.Series] = {}

        try:
            for epoch in range(self.config['num_epochs']):
                _, metric_eval = self._epoch()
                self._log()

                if self.progress['f1_eval'][-1] == max(self.progress['f1_eval']):
                    best_eval_metric = metric_eval
                    self.model.save_pretrained(self.config["export_path"])

        except KeyboardInterrupt:
            logging.warning('Warning: Training interrupted by user, skipping to evaluation if possible.')

        if best_eval_metric:
            self._evaluate(best_eval_metric)

    def _epoch(self) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
        time_begin: datetime = datetime.now()

        loss_train, f1_train, metric_train = self._step(self.dataset['train'], optimize=True)
        loss_eval, f1_eval, metric_eval = self._step(self.dataset['val'])

        self.progress['epoch'].append(len(self.progress['epoch']) + 1)
        self.progress['duration'].append(datetime.now() - time_begin)

        self.progress['loss_train'].append(loss_train)
        self.progress['loss_eval'].append(loss_eval)

        self.progress['f1_train'].append(f1_train)
        self.progress['f1_eval'].append(f1_eval)

        return metric_train, metric_eval

    def _step(self, dataset: Dataset, optimize: bool = False) -> Tuple[float, float, Dict[str, pd.Series]]:
        loss_value: float = 0.0
        metric = Metric(decoding_fn=dataset.decode_target_label)

        for batch in tqdm(
                DataLoader(
                    dataset,
                    shuffle=True,
                    drop_last=True,
                    batch_size=self.config['batch_size'],
                    collate_fn=self.collation_fn
                ),
                leave=False
        ):

            loss = self._forward(batch, metric)
            loss_value += loss.item()

            if optimize:
                self._optimize(loss)

            del loss

        return loss_value / (len(dataset) / self.config['batch_size']), metric.f_score(), metric.data

    def _forward(self, batch: dict, metric: Metric) -> torch.Tensor:
        predictions, loss = self.model(**batch)

        metric.add_observations(
            pd.Series(batch['labels'].cpu().numpy()),
            pd.Series(torch.argmax(predictions, dim=1).cpu().numpy())
        )

        return loss

    def _optimize(self, loss: torch.Tensor) -> None:
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

    def _evaluate(self, metric: Dict[str, pd.Series]) -> None:
        logging.info(f'[--- EVALUATION on max(f1_eval) ---]')
        metric = Metric(decoding_fn=self.dataset['train'].decode_target_label, **metric)
        metric.export(self.config["export_path"])
        logging.info(metric)

        (
            pd.DataFrame
            .from_records(self.progress, index=["epoch"])
            .to_csv(f'{self.config["export_path"]}/metric.train.csv')
        )

    def _log(self) -> None:
        logging.info((
            f'[@{self.progress["epoch"][-1]:03}]: \t'
            f'loss_train={self.progress["loss_train"][-1]:2.4f} \t'
            f'loss_eval={self.progress["loss_eval"][-1]:2.4f} \t'
            f'f1_train={self.progress["f1_train"][-1]:2.4f} \t'
            f'f1_eval={self.progress["f1_eval"][-1]:2.4f} \t'
            f'duration={self.progress["duration"][-1]}'
        ))
