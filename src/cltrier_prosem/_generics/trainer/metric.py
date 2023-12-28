from typing import Union, Dict
from dataclasses import dataclass, field

import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report


@dataclass
class Metric:
    decoding_fn: callable = lambda x: x
    golds: pd.Series = field(default_factory=pd.Series)
    preds: pd.Series = field(default_factory=pd.Series)

    def add_observations(self, golds: pd.Series, preds: pd.Series):
        self.golds = golds if self.golds.empty else pd.concat([self.golds, golds], ignore_index=True)
        self.preds = preds if self.preds.empty else pd.concat([self.preds, preds], ignore_index=True)

    def f_score(self, reduce: str = "weighted") -> float:
        return f1_score(self.golds, self.preds, average=reduce, zero_division=0.0)

    def accuracy(self) -> float:
        return accuracy_score(self.golds, self.preds)

    def cross_tabulation(self) -> pd.DataFrame:
        return pd.crosstab(
            self.golds.apply(self.decoding_fn),
            self.preds.apply(self.decoding_fn),
            rownames=['gold'],
            colnames=['pred']
        )

    def classification_report(self) -> pd.DataFrame:
        return pd.DataFrame(
            classification_report(
                self.golds.apply(self.decoding_fn),
                self.preds.apply(self.decoding_fn),
                zero_division=0.0,
                output_dict=True
            )
        ).T.drop('accuracy')

    def export(self, path: str) -> None:
        self.classification_report().to_csv(f'{path}/metric.classification_report.csv')
        self.cross_tabulation().to_csv(f'{path}/metric.cross_tabulation.csv')

    @property
    def data(self) -> Dict[str, pd.Series]:
        return {'golds': self.golds, 'preds': self.preds}

    def __repr__(self) -> Union[str, dict]:
        return self.classification_report().__repr__()
