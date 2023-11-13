import sys
from typing import List

import pandas as pd

from .classifier import Classifier
from .cli import CLI
from .pooler import Pooler


class Main(CLI):

    def load_classifier(self):
        return Classifier(
            in_size=self.encoder.dim,
            out_size=len(self.config['dataset']['label_classes']),
            **self.config['classifier']
        )

    def collate(self, batch: List[pd.DataFrame]) -> dict:
        pre_collated: dict = super().collate(batch)
        collated_data: pd.DataFrame = pd.concat(batch)

        if 'subword' in self.config['pooler']['form']:
            pre_collated['text'] = collated_data[self.config["dataset"]["text_column"]].tolist()
            pre_collated['span_idx'] = collated_data[self.config["pooler"]["span_column"]].tolist()

        return {
            'embeds': Pooler.batch_pool(
                pre_collated,
                form=self.config['pooler']['form'],
                encoder=self.encoder
            ),
            'labels': pre_collated['labels']
        }


if __name__ == '__main__':
    Main()()
