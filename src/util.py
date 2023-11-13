import argparse
import logging
from functools import wraps
from time import time
from typing import Union, Callable, Dict

import tomli
import torch


def setup_args_parser(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        'config',
        metavar='C',
        type=str,
        help='path to config.toml file'
    )
    parser.add_argument(
        '--debug',
        default=False,
        action=argparse.BooleanOptionalAction,
        help='enable debug level logging'
    )
    return parser, parser.parse_args()


def setup_logging(debug: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler()]
    )


def load_config(path: str) -> Dict[str, any]:
    return tomli.load(open(path, 'rb'))


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
