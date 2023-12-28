import logging as logger
import os
from typing import List, Tuple

import torch
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel, logging

from ..util import get_device, unpad, timing, model_memory_usage

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class EncoderConfig(BaseModel):
    model: str = "deepset/gbert-base"
    max_length: int = 512
    layers: List[int] = [-1]


class Encoder:

    @timing
    def __init__(self, config: EncoderConfig):
        logging.set_verbosity_error()
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        self.model = AutoModel.from_pretrained(
            config.model, output_hidden_states=True
        ).to(get_device())

        logger.info(self)

    @torch.no_grad()
    def __call__(
            self,
            batch: List[str],
            return_unpad: bool = True
    ) -> dict:
        token, encoding = self.tokenize(batch)
        embeds: torch.Tensor = self.forward(
            torch.tensor(encoding['input_ids'], device=get_device()).long(),
            torch.tensor(encoding['attention_mask'], device=get_device()).short()
        )

        return {
            'embeds': unpad(embeds, torch.tensor(encoding['attention_mask']).sum(1)) if return_unpad else embeds,
            'token': unpad(token, torch.tensor(encoding['attention_mask']).sum(1)) if return_unpad else token,
            **encoding
        }

    def tokenize(self, batch: List[str], padding: bool = True) -> Tuple[List[List[str]], dict]:
        encoding = self.tokenizer(
            batch,
            padding=padding,
            max_length=self.config.max_length,
            truncation=True,
            return_offsets_mapping=True
        )

        return (
            [self.ids_to_tokens(ids) for ids in encoding['input_ids']],
            encoding
        )

    @torch.no_grad()
    def forward(self, ids: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            self.model.forward(ids, masks).hidden_states[i]
            for i in self.config.layers
        ]).sum(0).squeeze()

    def ids_to_tokens(self, ids: torch.Tensor) -> List[str]:
        return self.tokenizer.convert_ids_to_tokens(ids)

    def ids_to_sent(self, ids: torch.Tensor) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @property
    def dim(self) -> int:
        return self.model.config.to_dict()['hidden_size']

    def __len__(self) -> int:
        return self.model.config.to_dict()['vocab_size']

    def __repr__(self):
        return (
            f'> Encoder Name: {self.model.config.__dict__["_name_or_path"]}\n'
            f'  Memory Usage: {model_memory_usage(self.model)}'
        )
