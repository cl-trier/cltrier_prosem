from typing import List

from pydantic import BaseModel

from ._generics.nn import EncoderConfig
from ._generics.trainer import TrainerConfig
from .pooler import PoolerConfig


class ClassifierConfig(BaseModel):
    hid_size: int = 512
    dropout: float = 0.2


class DatasetConfig(BaseModel):
    path: str
    text_column: str
    label_column: str
    label_classes: List[str]


class PipelineConfig(BaseModel):
    encoder: EncoderConfig
    dataset: DatasetConfig
    pooler: PoolerConfig
    classifier: ClassifierConfig
    trainer: TrainerConfig
