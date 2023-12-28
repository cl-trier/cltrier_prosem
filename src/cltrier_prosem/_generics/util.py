import logging
from functools import wraps
from time import time
from typing import Union, Callable

import torch


def get_device() -> str:
    return f'cuda' if torch.cuda.is_available() else 'cpu'


def unpad(
        padded: Union[list, torch.Tensor],
        length: Union[list, torch.Tensor]
) -> Union[list, torch.Tensor]:
    return [v[:n] for v, n in zip(padded, length)]


def timing(func: Callable):
    @wraps(func)
    def wrap(*args, **kwargs) -> any:
        start = time()
        result = func(*args, **kwargs)
        logging.info(f'> f({func.__name__}) took: {time() - start:2.4f} sec')

        return result

    return wrap


def model_memory_usage(model: torch.nn.Module) -> str:
    usage_in_byte: int = sum([
        sum([param.nelement() * param.element_size() for param in model.parameters()]),
        sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    ])

    return f'{usage_in_byte / (1024.0 * 1024.0):2.4f} MB'
