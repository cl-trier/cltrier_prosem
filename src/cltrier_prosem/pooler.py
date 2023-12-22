from typing import Literal, List, Tuple

import torch


class Pooler:
    extractions: dict = {
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

    @staticmethod
    def batch_pool(
            encoded_batch: dict,
            form: Literal[
                'cls', 'sent_mean',
                'subword_first', 'subword_last',
                'subword_mean', 'subword_min', 'subword_max'
            ] = 'cls'
    ):
        return torch.stack([
            Pooler.extractions[form](embed)
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
            emb_span_idx = Pooler._get_token_idx(mapping, span)
            yield embeds[emb_span_idx[0]: emb_span_idx[1] + 1]

    @staticmethod
    def _get_token_idx(offset_mapping: List[Tuple[int, int]], char_span: Tuple[int, int]):
        return (
            list(zip(*offset_mapping))[0].index(char_span[0]),
            next(emb_idx for emb_idx, char_idx in enumerate(list(zip(*offset_mapping))[1]) if char_idx >= char_span[1])
        )
