import logging
from dataclasses import dataclass

import pandas as pd
from torch.utils.data import Dataset as torchDataset


@dataclass
class Dataset(torchDataset):
    split: str
    data: pd.DataFrame
    data_label: str
    target_label: str

    def __post_init__(self):
        logging.info(self)

    def __getitem__(self, idx) -> pd.DataFrame:
        return self.data.iloc[[idx]]

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return (
            f'> Dataset: {self.split}\n'
            f'  Samples: {len(self)}'
        )
