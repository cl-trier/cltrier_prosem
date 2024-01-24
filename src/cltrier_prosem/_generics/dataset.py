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
        """
        Initialize the object after it has been deserialized. Logs the object using the logging module.
        """
        logging.info(self)

    def __getitem__(self, idx) -> pd.DataFrame:
        """
        Get a specific row of the dataframe by its index.

        Parameters:
            idx (int): The index of the row to retrieve.

        Returns:
            pd.DataFrame: The dataframe containing the selected row.
        """
        return self.data.iloc[[idx]]

    def __len__(self) -> int:
        """
        Return the length of the data as the data of samples.
        """
        return len(self.data)

    def __repr__(self) -> str:
        """
        Return a string representation of the dataset information including the label and the number of samples.

        Return:
            str: The string representation of the object.
        """
        return (
            f'> Dataset: {self.split}\n'
            f'  Samples: {len(self)}'
        )
