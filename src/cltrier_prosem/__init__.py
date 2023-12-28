"""ProSem - Probing and Classifying Semantic Spans"""

__version__ = "0.2.1"

import logging
import warnings
from typing import List

import pandas as pd
import torch

from ._config import PipelineConfig
from ._generics import Dataset, Trainer
from ._generics.nn import MultilayerPerceptron, Encoder
from ._generics.util import get_device
from .pooler import Pooler


class Pipeline:

    def __init__(self, config_dict: dict, debug: bool = False):
        logging.basicConfig(
            level=logging.DEBUG if debug else logging.INFO,
            format="%(message)s",
            handlers=[logging.StreamHandler()]
        )
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=UserWarning)

        logging.info('[--- SETUP ---]')
        self.config: PipelineConfig = PipelineConfig(**config_dict)
        logging.info(f'> Computation Device: {get_device()}')

        logging.info('[--- LOAD ENCODER ---]')
        self.encoder = Encoder(self.config.encoder)
        self.target_mapping = {
            class_name: n for n, class_name in
            enumerate(self.config.dataset.label_classes)
        }

        logging.info('[--- LOAD TRAINER ---]')
        self.trainer = Trainer(
            dataset={
                split: Dataset(
                    split=split,
                    data=pd.read_parquet(
                        f'{self.config.dataset.path}/{split}.parquet'
                    ),
                    data_label=self.config.dataset.text_column,
                    target_label=self.config.dataset.label_column,
                )
                for split in ['train', 'test']
            },
            model=self.load_classifier(),
            collation_fn=self.collate,
            label_decoding_fn=lambda x: {v: k for k, v in self.target_mapping.items()}.get(x),
            config=self.config.trainer
        )

    def __call__(self) -> None:
        logging.info('[--- RUN TRAINER ---]')
        self.trainer()

    def load_classifier(self) -> MultilayerPerceptron:
        return MultilayerPerceptron(
            in_size=self.encoder.dim,
            out_size=len(self.config.dataset.label_classes),
            **self.config.classifier.model_dump()
        )

    def collate(self, batch: List[pd.DataFrame]) -> dict:
        collated_data: pd.DataFrame = pd.concat(batch)

        return {
            'embeds': Pooler.batch_pool({
                'span_idx': collated_data[self.config.pooler.span_column].tolist(),
                **self.encoder(collated_data[self.config.dataset.text_column].tolist())
            }, form=self.config.pooler.form),
            'labels': torch.tensor(
                [
                    self.target_mapping[label] for label
                    in collated_data[self.config.dataset.label_column].tolist()
                ],
                dtype=torch.long,
                device=get_device()
            )
        }
