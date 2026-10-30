"""Second-stage reranking.

`FeatureReranker` is a transparent, dependency-free reranker that scores the *shape* of the
match between query and chunk, which first-stage retrievers ignore:

  coverage   fraction of distinct query terms present in the chunk
  bigram     fraction of adjacent query term pairs that also appear adjacent in the chunk
  proximity  how tightly the matched terms cluster (smallest window containing all of them)
  heading    overlap between query terms and the section heading

It is NOT a learned cross-encoder. It is a feature baseline whose benefit (or lack of one) is
measured by the evaluation harness rather than assumed. Implement `Reranker` to plug in a
cross-encoder; the harness will tell you whether it earns its latency.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Protocol

from docqa.schemas import Hit
from docqa.text import tokenize


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]: ...


def term_coverage(query_terms: Collection[str], chunk_terms: Collection[str]) -> float:
    qset = set(query_terms)
    if not qset:
        return 0.0
    return len(qset & set(chunk_terms)) / len(qset)


def _min_window(positions: dict[str, list[int]]) -> int:
    """Length of the smallest token window containing at least one occurrence of every term."""
    events = sorted((pos, term) for term, plist in positions.items() for pos in plist)
    needed = len(positions)
    counts: dict[str, int] = {}
    have = 0
    best = 10**9
    left = 0
    for pos_r, term_r in events:
        counts[term_r] = counts.get(term_r, 0) + 1
        if counts[term_r] == 1:
            have += 1
        while have == needed:
            best = min(best, pos_r - events[left][0] + 1)
            term_l = events[left][1]
            counts[term_l] -= 1
            if counts[term_l] == 0:
                have -= 1
            left += 1
    return best


def _proximity(query_terms: Sequence[str], chunk_tokens: Sequence[str]) -> float:
    wanted = set(query_terms)
    positions: dict[str, list[int]] = {}
    for i, tok in enumerate(chunk_tokens):
        if tok in wanted:
            positions.setdefault(tok, []).append(i)
    matched = len(positions)
    if matched == 0:
        return 0.0
    if matched == 1:
        return 1.0 / len(wanted)
    return matched / _min_window(positions) * (matched / len(wanted))


def _bigrams(tokens: Sequence[str]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:], strict=False))


class FeatureReranker:
    name = "feature"

    def __init__(self, fusion_weight: float = 0.35) -> None:
        if not 0.0 <= fusion_weight <= 1.0:
            raise ValueError("fusion_weight must be in [0, 1]")
        self.fusion_weight = fusion_weight

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        q_tokens = tokenize(query)
        if not q_tokens or not hits:
            return hits
        q_unique = list(dict.fromkeys(q_tokens))
        q_bigrams = _bigrams(q_tokens)
        max_rrf = max((h.signals.get("rrf", 0.0) for h in hits), default=0.0) or 1.0

        rescored: list[Hit] = []
        for hit in hits:
            c_tokens = tokenize(hit.chunk.index_text)
            coverage = term_coverage(q_unique, c_tokens)
            bigram = len(q_bigrams & _bigrams(c_tokens)) / len(q_bigrams) if q_bigrams else 0.0
            proximity = _proximity(q_unique, c_tokens)
            heading = term_coverage(q_unique, tokenize(hit.chunk.section))
            feature = 0.45 * coverage + 0.20 * bigram + 0.20 * proximity + 0.15 * heading
            fused = hit.signals.get("rrf", 0.0) / max_rrf
            final = self.fusion_weight * fused + (1 - self.fusion_weight) * feature
            signals = {
                **hit.signals,
                "coverage": round(coverage, 4),
                "bigram": round(bigram, 4),
                "proximity": round(proximity, 4),
                "heading": round(heading, 4),
                "rerank": round(final, 6),
            }
            rescored.append(hit.model_copy(update={"score": final, "signals": signals}))
        rescored.sort(key=lambda h: (-h.score, h.chunk.chunk_id))
        return rescored
