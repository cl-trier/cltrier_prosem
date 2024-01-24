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
        """
       Initialize the encoder with the provided EncoderConfig.

       Args:
           config (EncoderConfig): The configuration for the encoder.
       """
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
        """
        Tokenizes input batch and returns embeddings and tokens with optional padding removal.

        Args:
            batch (List[str]): List of input strings to be tokenized.
            return_unpad (bool, optional): Whether to remove padding from embeddings and tokens. Defaults to True.

        Returns:
            dict: A dictionary containing 'embeds', 'token', and **encoding.
        """
        token, encoding = self.tokenize(batch)
        embeds: torch.Tensor = self.forward(
            torch.tensor(encoding['input_ids'], device=get_device()).long(),
            torch.tensor(encoding['attention_mask'], device=get_device()).short()
        )

        def unpad_output(output: torch.tensor, mask: torch.tensor) -> torch.tensor:
            """
            Takes a tensor as input and returns an unpad output.
            Args:
                output (torch.tensor): The input torch.tensor.
                mask (torch.tensor): The mask torch.tensor.
            Returns:
                torch.tensor: The unpad output.
            """
            return unpad(output, torch.tensor(mask).sum(1)) if return_unpad else output

        return {
            'embeds': unpad_output(embeds, encoding['attention_mask']),
            'token': unpad_output(token, encoding['attention_mask']),
            **encoding
        }

    def tokenize(self, batch: List[str], padding: bool = True) -> Tuple[List[List[str]], dict]:
        """
        Tokenizes a batch of strings and returns the tokenized batch along with the encoding.

        Args:
            batch (List[str]): The list of strings to tokenize.
            padding (bool, optional): Whether to pad the sequences to the maximum length. Defaults to True.

        Returns:
            Tuple[List[List[str]], dict]: The tokenized batch and the encoding.
        """
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
        """
        Perform a forward pass through the model and return the aggregated hidden states.

        Args:
            ids (torch.Tensor): The input tensor for token ids.
            masks (torch.Tensor): The input tensor for attention masks.

        Returns:
            torch.Tensor: The aggregated hidden states obtained from the model's forward pass.
        """
        return torch.stack([
            self.model.forward(ids, masks).hidden_states[i]
            for i in self.config.layers
        ]).sum(0).squeeze()

    def ids_to_tokens(self, ids: torch.Tensor) -> List[str]:
        """
        Convert the input token IDs to a list of token strings using the internal tokenizer.

        Args:
            ids (torch.Tensor): The input token IDs to be converted to tokens.

        Returns:
            List[str]: A list of token strings corresponding to the input token IDs.
        """
        return self.tokenizer.convert_ids_to_tokens(ids)

    def ids_to_sent(self, ids: torch.Tensor) -> str:
        """
        Convert the input tensor of token IDs to a string using the internal tokenizers decode method.

        Args:
            ids (torch.Tensor): The input tensor of token IDs.

        Returns:
            str: The decoded string output.
        """
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @property
    def dim(self) -> int:
        """
        Return the dimension of the model.
        """
        return self.model.config.to_dict()['hidden_size']

    def __len__(self) -> int:
        """
        Return the length of the object based on the vocabulary size.
        """
        return self.model.config.to_dict()['vocab_size']

    def __repr__(self) -> str:
        """
        Return a string representation of the encoder including memory usage.
        """
        return (
            f'> Encoder Name: {self.model.config.__dict__["_name_or_path"]}\n'
            f'  Memory Usage: {model_memory_usage(self.model)}'
        )
