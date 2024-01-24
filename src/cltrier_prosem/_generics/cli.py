import argparse
from abc import ABC, abstractmethod

import tomli


class CLI(ABC):

    def __init__(self, name: str):
        """
        Initialize the class with the given name.

        Parameters:
            name (str): the name to be used as the description for the argparse parser
        """
        self.parser = argparse.ArgumentParser(description=name)
        self.parser.add_argument(
            'config',
            metavar='C',
            type=str,
            help='path to config.toml file'
        )
        self.parser.add_argument(
            '--debug',
            default=False,
            action=argparse.BooleanOptionalAction,
            help='enable debug level logging'
        )

        self.config = tomli.load(open(self.parser.parse_args().config, 'rb'))
        self.debug = self.parser.parse_args().debug

    @abstractmethod
    def __call__(self):
        """
        A method that represents the callable behavior of the object.
        """
        pass
