import logging
from dataclasses import dataclass, field
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

    progress: Progress = field(default_factory=Progress)

    def __post_init__(self):
        """
        Initialize data loaders and load optimizer for the model.
        """
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
        """
        Execute the training process for a specified number of epochs.
        If the training process is interrupted by the user,
        it will skip to the evaluation step if possible.
        """
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
        """
        Method to perform a single epoch of training and testing, and record the progress.
        """
        time_begin: datetime = datetime.now()

        self.progress.append_record(
            epoch=len(self.progress.epoch) + 1,
            duration=datetime.now() - time_begin,
            train_results=self._step(self.data_loader['train'], optimize=True),
            test_results=self._step(self.data_loader['test'])
        )

    def _step(self, data_loader: DataLoader, optimize: bool = False) -> Tuple[float, float, Dict[str, pd.Series]]:
        """
        Perform a step of the training process.

        Args:
            data_loader (DataLoader): The data loader for the training data.
            optimize (bool, optional): Whether to optimize the model. Defaults to False.

        Returns:
            Tuple[float, float, Dict[str, pd.Series]]: A tuple containing the average loss value, the F score metric,
            and a dictionary of metric data.
        """
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
        """
        Compute the forward pass through the model and update the metric with observations.

        Args:
            batch (dict): The input data batch.
            metric (Metric): The metric object for tracking performance.

        Returns:
            torch.Tensor: The loss value from the forward pass.
        """
        predictions, loss = self.model(**batch)

        metric.add_observations(
            pd.Series(batch['labels'].cpu().numpy()),
            pd.Series(torch.argmax(predictions, dim=1).cpu().numpy())
        )

        return loss

    def _optimize(self, loss: torch.Tensor) -> None:
        """
        Optimize the model by performing a backward pass to compute gradients,
        then taking a step with the optimizer and zeroing the gradients.

        Args:
            loss (torch.Tensor): The loss value to perform backpropagation.
        """
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _evaluate(self) -> None:
        """
        Evaluate the model using the max value of f1_test and export the metric to the specified export_path.
        """
        logging.info(f'[--- EVALUATION on max(f1_test) ---]')
        metric = Metric(
            decoding_fn=self.label_decoding_fn,
            **self.progress.metric_test[self.progress.max_record_id]
        )
        metric.export(self.config.export_path)
        logging.info(metric)
