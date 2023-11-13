"""ToDo!"""

__version__ = "0.1.5"

import logging
from typing import List

import pandas as pd
import torch

from .classifier import Classifier
from .dataset import Dataset
from .encoder import Encoder
from .metric import Metric
from .pooler import Pooler
from .trainer import Trainer
from .util import setup_args_parser, setup_logging, load_config, get_device


class Pipeline:

    def __init__(self, config: dict):
        self.config: dict = config

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
                for split in ['train', 'test']
            },
            model=self.load_classifier(),
            collation_fn=self.collate,
            config=self.config['trainer']
        )

    def __call__(self) -> None:
        logging.info('[--- RUN TRAINER ---]')
        self.trainer()

    def load_classifier(self) -> Classifier:
        return Classifier(
            in_size=self.encoder.dim,
            out_size=len(self.config['dataset']['label_classes']),
            **self.config['classifier']
        )

    def collate(self, batch: List[pd.DataFrame]) -> dict:
        collated_data: pd.DataFrame = pd.concat(batch)

        pre_collated: dict = {
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

        if 'subword' in self.config['pooler']['form']:
            pre_collated['text'] = collated_data[self.config["dataset"]["text_column"]].tolist()
            pre_collated['span_idx'] = collated_data[self.config["pooler"]["span_column"]].tolist()

        return {
            'embeds': Pooler.batch_pool(
                pre_collated,
                form=self.config['pooler']['form'],
                encoder=self.encoder
            ),
            'labels': pre_collated['labels']
        }
