from . import Pipeline
from .util import setup_args_parser


class CLI:

    def __init__(self):
        parser, args = setup_args_parser('ProSem - Span Classification')
        self.pipeline = Pipeline(args.config)

    def __call__(self) -> None:
        self.pipeline()


if __name__ == '__main__':
    CLI()()
