from typing import Literal, List, Tuple

import pandas as pd
import torch
from pydantic import BaseModel

POOL_FORM_TYPE = Literal[
    'cls', 'sent_mean',
    'subword_first', 'subword_last',
    'subword_mean', 'subword_min', 'subword_max'
]

POOL_FORM_FNS: dict = {
    # sentence based
    'cls': lambda x: x[0],
    'sent_mean': lambda x: torch.mean(x[1:-1], dim=0),

    # word based, positional extraction
    'subword_first': lambda x: x[0],
    'subword_last': lambda x: x[-1],

    # word based, arithmetic extraction
    'subword_mean': lambda x: torch.mean(x, dim=0),
    'subword_min': lambda x: torch.min(x, dim=0)[0],
    'subword_max': lambda x: torch.max(x, dim=0)[0]
}


class PoolerConfig(BaseModel):
    form: POOL_FORM_TYPE = 'cls'
    span_columns: List[str]


class Pooler:

    @staticmethod
    def pool_multi(
            collated_data: pd.DataFrame,
            encoded_batch: dict,
            span_columns: List[str],
            form: POOL_FORM_TYPE
    ) -> List[torch.tensor]:
        return [
            Pooler.pool_batch({
                'span_idx': collated_data[span_column].tolist(),
                **encoded_batch
            },
                form=form
            )
            for span_column in span_columns
        ]

    @staticmethod
    def pool_batch(
            encoded_batch: dict,
            form: POOL_FORM_TYPE = 'cls'
    ):
        return torch.stack([
            POOL_FORM_FNS[form](embed)
            for embed in (
                encoded_batch['embeds']
                if form in ['cls', 'sent_mean'] else
                Pooler._extract_embed_spans(encoded_batch)
            )
        ])

    @staticmethod
    def _extract_embed_spans(encoded_batch: dict):
        for span, mapping, embeds in zip(
                encoded_batch['span_idx'],
                encoded_batch['offset_mapping'],
                encoded_batch['embeds']
        ):
            emb_span_idx = Pooler._get_token_idx(mapping[1:embeds.size(dim=0) - 1], span)
            yield embeds[emb_span_idx[0]: emb_span_idx[1] + 1]

    @staticmethod
    def _get_token_idx(mapping: List[Tuple[int, int]], c_span: Tuple[int, int]) -> Tuple[int, int]:
        prep_map: callable = lambda pos: list(enumerate(list(zip(*mapping))[pos]))

        span: Tuple[int, int] = (
            next(eid for eid, cid in reversed(prep_map(0)) if cid <= c_span[0]),
            next(eid for eid, cid in prep_map(1) if cid >= c_span[1])
        )

        return span if span[0] <= span[1] else (span[1], span[0])
