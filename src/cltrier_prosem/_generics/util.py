import logging
from functools import wraps
from time import time
from typing import Union, Callable

import torch


def get_device() -> str:
    """
    Return the computation device as a string based on the availability of CUDA.
    """
    return f'cuda' if torch.cuda.is_available() else 'cpu'


def unpad(
        padded: Union[list, torch.Tensor],
        length: Union[list, torch.Tensor]
) -> Union[list, torch.Tensor]:
    """
    Removes padding from the input sequences based on the specified lengths.

    Args:
        padded (Union[list, torch.Tensor]): The padded input sequences.
        length (Union[list, torch.Tensor]): The lengths of the sequences.

    Returns:
        Union[list, torch.Tensor]: The unpadded sequences.
    """
    return [v[:n] for v, n in zip(padded, length)]


def timing(func: Callable):
    """
    Decorator function to measure the execution time of the input function.

    Args:
        func (Callable): The function to be measured.

    Returns:
        any: The result of the input function.
    """

    @wraps(func)
    def wrap(*args, **kwargs) -> any:
        """
        Decorator function that wraps the input function, measures its execution time,
        logs the result, and returns the result.
        """
        start = time()
        result = func(*args, **kwargs)
        logging.info(f'> f({func.__name__}) took: {time() - start:2.4f} sec')

        return result

    return wrap


def model_memory_usage(model: torch.nn.Module) -> str:
    """
    Calculate the memory usage of the input model in megabytes.

    Args:
        model (torch.nn.Module): The input model for which memory usage needs to be calculated.

    Returns:
        str: A string representing the memory usage of the model in megabytes.
    """
    usage_in_byte: int = sum([
        sum([param.nelement() * param.element_size() for param in model.parameters()]),
        sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    ])

    return f'{usage_in_byte / (1024.0 * 1024.0):2.4f} MB'
