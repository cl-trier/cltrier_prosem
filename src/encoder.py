import logging as logger
import os
from typing import List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModel, logging

from .util import get_device, unpad, timing, model_memory_usage

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Encoder:

    @timing
    def __init__(
            self,
            model: str = "deepset/gbert-base",
            max_length: int = 512,
            layers: Optional[List[int]] = None,
    ):
        logging.set_verbosity_error()

        if layers is None:
            layers = [-1]

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model, output_hidden_states=True).to(get_device())

        self.max_length = max_length
        self.layers = layers

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
            truncation=True,
            max_length=self.max_length
        )

        return (
            [self.ids_to_tokens(ids) for ids in encoding['input_ids']],
            encoding
        )

    @torch.no_grad()
    def forward(self, ids: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            self.model.forward(ids, masks).hidden_states[i]
            for i in self.layers
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
