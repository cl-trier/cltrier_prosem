"""ProSem - Probing and Classifying Semantic Spans"""

__version__ = "0.4.1"

import logging
import warnings
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

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

        logging.info('[--- LOAD DATA ---]')
        self.dataset: Dict[str, Dataset] = {
            split: Dataset(
                split=split,
                data=pd.read_parquet(
                    f'{self.config.dataset.path}/{split}.parquet'
                ),
                data_label=self.config.dataset.text_column,
                target_label=self.config.dataset.label_column,
            )
            for split in ['train', 'test']
        }

        logging.info('[--- EMBED DATA ---]')
        self.embed_text()

        logging.info('[--- LOAD TRAINER ---]')
        self.trainer = Trainer(
            dataset=self.dataset,
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
            in_size=self.encoder.dim * len(self.config.pooler.span_columns),
            out_size=len(self.config.dataset.label_classes),
            **self.config.classifier.model_dump()
        )

    def embed_text(self) -> None:
        for split in self.dataset.values():
            split_embeds = []
            for _, batch in tqdm(split.data.groupby(np.arange(len(split.data)) // self.config.trainer.batch_size)):
                embeds: dict = self.encoder(batch[self.config.dataset.text_column].tolist())
                pooled_embeds: List[torch.tensor] = Pooler.pool_multi(
                    batch, embeds,
                    self.config.pooler.span_columns,
                    self.config.pooler.form
                )
                pooled_collated_embeds = torch.cat(
                    pooled_embeds, dim=1 if len(pooled_embeds[0]) != 1 else 0
                )
                split_embeds.extend([embed for embed in pooled_collated_embeds])

            split.data['embeds'] = [np.array(embed.numpy()) for embed in split_embeds]

            if True:
                split.data.to_parquet(f'{self.config.trainer.export_path}/embeds.{split.split}.parquet')
                split.data['embeds'] = split_embeds

    def encode_labels(self, labels: pd.Series) -> torch.tensor:
        return torch.tensor(
            [self.target_mapping[label] for label in labels.tolist()],
            dtype=torch.long,
            device=get_device()
        )

    def collate(self, batch: List[pd.DataFrame]) -> dict:
        collated_data: pd.DataFrame = pd.concat(batch)

        return {
            'inputs': torch.stack(collated_data.embeds.tolist()),
            'labels': self.encode_labels(collated_data[self.config.dataset.label_column])
        }
