import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Dict, Callable

import pandas as pd
import torch
from pydantic import BaseModel
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .metric import Metric
from .progress import Progress


class TrainerConfig(BaseModel):
    num_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-3

    export_path: str


@dataclass
class Trainer:
    dataset: Dict[str, Dataset]
    model: torch.nn.Module

    collation_fn: Callable
    label_decoding_fn: Callable

    config: TrainerConfig

    progress: Progress = Progress()

    def __post_init__(self):
        # create data loaders
        self.data_loader: Dict[str, DataLoader] = {
            label: DataLoader(
                dataset,
                shuffle=True,
                drop_last=True,
                batch_size=self.config.batch_size,
                collate_fn=self.collation_fn
            )
            for label, dataset in self.dataset.items()
        }

        #  load optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)

    def __call__(self) -> None:
        try:
            for epoch in range(self.config.num_epochs):
                self._epoch()
                self.progress.log()

                if self.progress.last_is_best:
                    self.model.save_pretrained(self.config.export_path)

        except KeyboardInterrupt:
            logging.warning('Warning: Training interrupted by user, skipping to evaluation if possible.')

        if self.progress:
            self._evaluate()
            self.progress.export(f'{self.config.export_path}/metric.train')

    def _epoch(self) -> None:
        time_begin: datetime = datetime.now()

        self.progress.append_record(
            epoch=len(self.progress.epoch) + 1,
            duration=datetime.now() - time_begin,
            train_results=self._step(self.data_loader['train'], optimize=True),
            test_results=self._step(self.data_loader['test'])
        )

    def _step(self, data_loader: DataLoader, optimize: bool = False) -> Tuple[float, float, Dict[str, pd.Series]]:
        loss_value: float = 0.0
        metric = Metric(decoding_fn=self.label_decoding_fn)

        for batch in tqdm(data_loader, leave=False):
            loss = self._forward(batch, metric)
            loss_value += loss.item()

            if optimize:
                self._optimize(loss)

            del loss

        return loss_value / len(data_loader), metric.f_score(), metric.data

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
        self.optimizer.zero_grad()

    def _evaluate(self) -> None:
        logging.info(f'[--- EVALUATION on max(f1_test) ---]')
        metric = Metric(
            decoding_fn=self.label_decoding_fn,
            **self.progress.metric_test[self.progress.max_record_id]
        )
        metric.export(self.config.export_path)
        logging.info(metric)
