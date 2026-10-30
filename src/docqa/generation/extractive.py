"""Extractive generator: answers by quoting the most relevant sentences of the top chunks.

Needs no LLM, so it is free, deterministic and cannot hallucinate (every word comes from a
cited source). It is also the reference generator the offline evaluation runs against. Use
`AnthropicGenerator` when you want fluent, synthesised answers.
"""

from __future__ import annotations

import re

from docqa.generation.base import SUPPORT_WINDOW, abstain, compute_support
from docqa.retrieval.rerank import term_coverage
from docqa.schemas import Answer, Citation, Hit
from docqa.text import tokenize

_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_LEADING_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")


def _sentences(text: str) -> list[str]:
    cleaned = (_LEADING_BULLET.sub("", s).strip() for s in _SENTENCE.split(text))
    return [s for s in cleaned if len(s) > 3 and not s.startswith("#")]


class ExtractiveGenerator:
    name = "extractive"

    def __init__(self, min_support: float = 0.5, max_sentences: int = 2) -> None:
        self.min_support = min_support
        self.max_sentences = max_sentences

    def generate(self, question: str, hits: list[Hit]) -> Answer:
        support = compute_support(question, hits)
        if support < self.min_support:
            return abstain(support, self.name)

        q_terms = set(tokenize(question))
        scored: list[tuple[float, int, int, str]] = []
        for hit_idx, hit in enumerate(hits[:SUPPORT_WINDOW]):
            for sent_idx, sentence in enumerate(_sentences(hit.chunk.text)):
                score = term_coverage(q_terms, tokenize(sentence)) - 0.01 * hit_idx
                scored.append((score, hit_idx, sent_idx, sentence))
        scored.sort(key=lambda t: (-t[0], t[1], t[2]))
        chosen = scored[: self.max_sentences]
        chosen.sort(key=lambda t: (t[1], t[2]))  # read in source order

        parts: list[str] = []
        citations: list[Citation] = []
        numbering: dict[int, int] = {}
        for _, hit_idx, _, sentence in chosen:
            if hit_idx not in numbering:
                numbering[hit_idx] = len(numbering) + 1
                hit = hits[hit_idx]
                citations.append(
                    Citation(
                        index=numbering[hit_idx],
                        chunk_id=hit.chunk.chunk_id,
                        doc_id=hit.chunk.doc_id,
                        section=hit.chunk.section,
                        quote=sentence,
                    )
                )
            parts.append(f"{sentence} [{numbering[hit_idx]}]")
        return Answer(
            text=" ".join(parts),
            abstained=False,
            support=round(support, 4),
            citations=citations,
            generator=self.name,
        )
