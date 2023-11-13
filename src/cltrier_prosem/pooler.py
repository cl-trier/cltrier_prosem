from typing import Literal, List

import numpy as np
import torch

from .encoder import Encoder
from .util import get_device


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
            ] = 'cls',
            encoder: Encoder = None
    ):

        if form in ['cls', 'sent_mean']:
            return torch.stack([
                Pooler.extractions[form](embed)
                for embed in encoded_batch['embeds']
            ])

        else:
            span_text: List[str] = Pooler._extract_span_text(encoded_batch['text'], encoded_batch['span_idx'])
            span_token, _ = encoder.tokenize(span_text, padding=False)

            matching_embeds = Pooler._slice_span_matching_embeds(
                span_embeds=Pooler._extract_token_matching_embeds(
                    sent_embeds=encoded_batch['embeds'],
                    sent_token=encoded_batch['token'],
                    span_token=span_token
                ),
                span_position=Pooler._calculate_span_position(
                    span_frequency=Pooler._count_span_frequency(span_text, encoded_batch['text']),
                    span_text=span_text,
                    span_idx=encoded_batch['span_idx'],
                    text=encoded_batch['text']
                ),
                span_token=span_token
            )

            return torch.stack([Pooler.extractions[form](embed) for embed in matching_embeds])

    @staticmethod
    def _extract_span_text(text: List[str], spans: List[np.ndarray]) -> List[str]:
        return [
            ' '.join(sent.split()[slice(*span.astype(int).tolist())])
            for sent, span in zip(text, spans)
        ]

    @staticmethod
    def _count_span_frequency(span: List[str], text: List[str]) -> List[int]:
        return [sent.count(span) for span, sent in zip(span, text)]

    @staticmethod
    def _calculate_span_position(
            span_frequency: List[int],
            span_text: List[str],
            span_idx: List[int],
            text: List[str]
    ) -> List[int]:

        return [
            Pooler._find_idx(tx, span, idx[0]) if freq > 1 else -1
            for freq, span, idx, tx in zip(span_frequency, span_text, span_idx, text)
        ]

    @staticmethod
    def _find_idx(tx: str, span: str, idx: str | int) -> int:
        try:
            return [i for i in range(len(tx.split())) if tx.split()[i] == span].index(int(idx))

        except ValueError:
            return -1

    @staticmethod
    def _extract_token_matching_embeds(
            sent_embeds: List[torch.Tensor],
            sent_token: List[List[str]],
            span_token: List[List[str]],
    ) -> List[torch.Tensor]:
        return [
            embeds[torch.tensor(
                [tok in span[1:-1] for tok in token],
                device=get_device()
            ), :]
            for embeds, token, span in zip(
                sent_embeds,
                sent_token,
                span_token,
            )
        ]

    @staticmethod
    def _slice_span_matching_embeds(
            span_embeds: List[torch.Tensor],
            span_position: List[int],
            span_token: List[List[str]]
    ) -> List[torch.Tensor]:
        return [
            embed[position * size: position * size + size] if position != -1 else embed
            for embed, position, size in zip(
                span_embeds,
                span_position,
                [len(tok) - 2 for tok in span_token]
            )
        ]
