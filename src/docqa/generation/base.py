"""Generator interface and the shared "is the evidence good enough?" gate.

`support` is the best query-term coverage achieved by any of the top chunks. When it falls
below `min_support`, every generator abstains instead of answering: an honest "not found" is
cheaper than a confident hallucination, and it also avoids paying for an LLM call.
"""

from __future__ import annotations

from typing import Protocol

from docqa.retrieval.rerank import term_coverage
from docqa.schemas import Answer, Hit
from docqa.text import tokenize

ABSTAIN_TEXT = "I could not find this in the indexed documents."
SUPPORT_WINDOW = 3  # how many top hits are inspected for evidence


class AnswerGenerator(Protocol):
    name: str

    def generate(self, question: str, hits: list[Hit]) -> Answer: ...


def compute_support(question: str, hits: list[Hit]) -> float:
    q_terms = set(tokenize(question))
    if not q_terms or not hits:
        return 0.0
    return max(
        term_coverage(q_terms, tokenize(h.chunk.index_text)) for h in hits[:SUPPORT_WINDOW]
    )


def abstain(support: float, generator: str) -> Answer:
    return Answer(text=ABSTAIN_TEXT, abstained=True, support=round(support, 4), generator=generator)
