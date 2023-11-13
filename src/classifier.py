import logging

import torch
import torch.nn as nn

from .tscloss import TSCLoss
from .util import get_device, model_memory_usage


class Classifier(nn.Module):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            hid_size: int = 32,
            dropout: float = 0.2,
            loss_fn: str = 'CrossEntropyLoss'
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_size, hid_size),
            nn.Dropout(p=dropout),
            nn.LeakyReLU(),
            nn.Linear(hid_size, out_size)
        ).to(get_device())

        if loss_fn == 'TSCLoss':
            self.loss = TSCLoss()

        else:
            self.loss = torch.nn.CrossEntropyLoss()

        logging.info(self)

    def __call__(self, embeds: torch.Tensor, labels: torch.Tensor):
        pred = self.predict(embeds)
        return pred, self.loss(pred, labels.view(-1))

    def predict(self, embeds: torch.Tensor) -> torch.Tensor:
        return self.model(embeds)

    def save_pretrained(self, path: str) -> None:
        torch.save({"state_dict": self.state_dict()}, f'{path}/model')

    def __len__(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f'> Model: Multilayer perceptron\n'
            f'  Memory Usage: {model_memory_usage(self.model)}'
            f' {self.model.__repr__()[:-1].replace("Sequential(", "")}'
        )
