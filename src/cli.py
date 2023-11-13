import logging
from typing import List

import pandas as pd
import torch

from .classifier import Classifier
from .encoder import Encoder
from .dataset import Dataset
from .trainer import Trainer
from .util import setup_args_parser, setup_logging, load_config, get_device


class CLI:

    def __init__(self):
        parser, args = setup_args_parser('ProSem - Span Classification')

        setup_logging(args.debug)
        self.config: dict = load_config(args.config)

        logging.info('[--- LOAD ENCODER ---]')
        self.encoder = Encoder(**self.config['encoder'])
        self.target_mapping = {
            class_name: n for n, class_name in
            enumerate(self.config["dataset"]["label_classes"])
        }

        logging.info('[--- LOAD TRAINER ---]')
        self.trainer = Trainer(
            dataset={
                split: Dataset(
                    split=split,
                    data=pd.read_parquet(
                        f'{self.config["dataset"]["path"]}/{split}.parquet'
                    ),
                    data_label=self.config["dataset"]["text_column"],
                    target_label=self.config["dataset"]["label_column"],
                    target_mapping=self.target_mapping
                )
                for split in ['train', 'val']
            },
            model=self.load_classifier(),
            collation_fn=self.collate,
            config=self.config['trainer']
        )

    def __call__(self) -> None:
        logging.info('[--- RUN TRAINER ---]')
        self.trainer()

    def load_classifier(self) -> Classifier:
        pass

    def collate(self, batch: List[pd.DataFrame]) -> dict:
        collated_data: pd.DataFrame = pd.concat(batch)

        return {
            'labels': torch.tensor(
                [
                    self.target_mapping[label] for label
                    in collated_data[self.config["dataset"]["label_column"]].tolist()
                ],
                dtype=torch.long,
                device=get_device()
            ),
            **self.encoder(collated_data[self.config["dataset"]["text_column"]].tolist())
        }
