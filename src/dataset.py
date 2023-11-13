import logging
from dataclasses import dataclass
from typing import Dict, Union

import pandas as pd
from torch.utils.data import Dataset as torchDataset


@dataclass
class Dataset(torchDataset):
    split: str
    data: pd.DataFrame
    data_label: str
    target_label: str
    target_mapping: Dict[str, int]

    def __post_init__(self):
        logging.info(self)

    def encode_target_label(self, label: Union[str, bool]) -> int:
        return self.target_mapping.get(str(label))

    def decode_target_label(self, label: int) -> str:
        return {v: k for k, v in self.target_mapping.items()}.get(label)

    def __getitem__(self, idx) -> pd.DataFrame:
        return self.data.iloc[[idx]]

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return (
            f'> Dataset: {self.split}\n'
            f'  Samples: {len(self)}'
        )
