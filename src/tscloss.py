import torch
import torch.nn as nn

from .util import get_device


class TSCLoss(nn.Module):
    """
    Temporal Segmented Classification Loss
    """

    def __init__(self, falloff: str = 'kernel'):
        super(TSCLoss, self).__init__()

        self.falloff: str = falloff

    def forward(self, inputs, targets):
        loss: torch.Tensor = torch.tensor(0)

        if self.falloff == 'sinusoidal':
            loss = TSCLoss._generate_sinus_decay(targets, inputs.size()[1]) - inputs

        if self.falloff == 'kernel':
            loss = TSCLoss._generate_smoothing_kernel_decay(targets, inputs.size()[1]) - inputs

        return torch.mean(torch.pow(loss, 2))

    @staticmethod
    def _generate_smoothing_kernel_decay(targets: torch.Tensor, target_length: int, sigma: float = 4.0):
        def get_kernel(peak_index: int):
            kernel = torch.exp(
                -((torch.arange(0, target_length, dtype=torch.float32, device=get_device()) - peak_index) ** 2)
                / (2 * sigma ** 2)
            )
            kernel = kernel / kernel.sum()

            return kernel

        return torch.stack([get_kernel(tgt) for tgt in targets])

    @staticmethod
    def _generate_sinus_decay(
            targets: torch.Tensor,
            target_length: int,
            fall_off_minimum: float = 0.0,
            fall_off_power: int = 4,
    ) -> torch.Tensor:
        step_size = torch.pi / target_length

        return torch.pow(
            torch.maximum(
                torch.sin(
                    torch.stack([
                        torch.arange(
                            step_size / 2 + step_size * (target_length / 2 - tgt.item()),
                            torch.pi + step_size / 2 + step_size * (target_length / 2 - tgt.item()),
                            step_size,
                            device=get_device()
                        ) for tgt in targets
                    ])
                ),
                torch.tensor(fall_off_minimum)
            ),
            fall_off_power
        )
