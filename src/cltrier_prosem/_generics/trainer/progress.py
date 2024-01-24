import logging
from datetime import timedelta
from typing import List, Tuple

import pandas as pd
from pydantic import BaseModel, computed_field


class Progress(BaseModel):
    epoch: List[int] = []
    duration: List[timedelta] = []

    loss_train: List[float] = []
    loss_test: List[float] = []

    f1_train: List[float] = []
    f1_test: List[float] = []

    metric_train: List[dict] = []
    metric_test: List[dict] = []

    def append_record(
            self,
            epoch: int,
            duration: timedelta,
            train_results: Tuple[float, float, dict],
            test_results: Tuple[float, float, dict]
    ):
        """
        Append a record of the epoch, duration, train results, and test results to their respective lists.

        Args:
            epoch (int): The epoch number.
            duration (timedelta): The duration of the training.
            train_results (Tuple[float, float, dict]): The results of the training.
            test_results (Tuple[float, float, dict]): The results of the testing.

        """
        self.epoch.append(epoch)
        self.duration.append(duration)

        self.loss_train.append(train_results[0])
        self.loss_test.append(test_results[0])

        self.f1_train.append(train_results[1])
        self.f1_test.append(test_results[1])

        self.metric_train.append(train_results[2])
        self.metric_test.append(test_results[2])

    @computed_field
    @property
    def max_record_id(self) -> int:
        """
        Returns the maximum record id as an integer.
        """
        if not self.f1_test:
            return -1

        return self.f1_test.index(max(self.f1_test))

    @computed_field
    @property
    def last_is_best(self) -> bool:
        """
        Returns a boolean indicating whether the last element in f1_test is the maximum.
        """
        if not self.f1_test:
            return False

        return self.f1_test[-1] == max(self.f1_test)

    def log(self) -> None:
        """
        Logs the training and testing metrics, including loss and F1 scores, along with the epoch number and duration.
        """
        logging.info((
            f'[@{self.epoch[-1]:03}]: \t'
            f'loss_train={self.loss_train[-1]:2.4f} \t'
            f'loss_test={self.loss_test[-1]:2.4f} \t'
            f'f1_train={self.f1_train[-1]:2.4f} \t'
            f'f1_test={self.f1_test[-1]:2.4f} \t'
            f'duration={self.duration[-1]}'
        ))

    def export(self, path: str) -> None:
        """
        Exports the record dump to a CSV file at the specified path.

        Args:
            path (str): The file path to export the CSV.
        """
        (
            pd.DataFrame
            .from_records(self.model_dump(
                exclude={'metric_train', 'metric_test'}
            ), index=['epoch'])
            .to_csv(f'{path}.csv')
        )
