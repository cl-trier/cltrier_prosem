import logging
from typing import Tuple

import torch
import torch.nn as nn

from ..util import get_device, model_memory_usage


class MultilayerPerceptron(nn.Module):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            hid_size: int,
            dropout: float
    ):
        """
        Initialize the multilayer perceptron with the specified input, output, and hidden layer sizes,
        as well as the dropout rate.

        Args:
            in_size (int): The size of the input layer.
            out_size (int): The size of the output layer.
            hid_size (int): The size of the hidden layer.
            dropout (float): The dropout rate.

        Returns:
            None
        """
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_size, hid_size),
            nn.Dropout(p=dropout),
            nn.LeakyReLU(),
            nn.Linear(hid_size, out_size)
        ).to(get_device())

        self.loss = torch.nn.CrossEntropyLoss()
        logging.info(self)

    def __call__(self, inputs: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Call method to make predictions and calculate loss.

        Args:
            inputs (torch.Tensor): The input tensor for making predictions.
            labels (torch.Tensor): The target labels for the predictions.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The predictions and the calculated loss.
        """
        pred = self.predict(inputs)
        return pred, self.loss(pred, labels.view(-1))

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Predicts the output using provided inputs.

        Args:
            inputs (torch.Tensor): The input tensor for prediction.

        Returns:
            torch.Tensor: The predicted output.
        """
        return self.model(inputs)

    def save_pretrained(self, path: str) -> None:
        """
        Save the pretrained model to the specified path.

        Args:
            path (str): The path where the model will be saved.

        Returns:
            None
        """
        torch.save({"state_dict": self.state_dict()}, f'{path}/model.bin')

    def __len__(self) -> int:
        """
        Return the length as the trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        """
        Return the string representation of the multilayer perceptron model including memory usage.
        """
        return (
            f'> Model: Multilayer perceptron\n'
            f'  Memory Usage: {model_memory_usage(self.model)}'
            f' {self.model.__repr__()[:-1].replace("Sequential(", "")}'
        )
