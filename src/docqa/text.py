"""Text normalisation shared by every component that touches tokens.

Using one tokenizer for chunk sizing, BM25, the dense embedder and the reranker keeps
the signals consistent: a term that BM25 matches is a term the reranker can see.

The stemmer is deliberately minimal (suffix stripping, no dictionary). It exists to make
"rotate / rotated / rotation-free" style variants meet in the index without pulling in a
heavyweight NLP dependency. Swap it for a real stemmer if your language needs it.
"""

from __future__ import annotations

import re
import unicodedata

_WORD = re.compile(r"[a-z0-9]+")
_WHITESPACE_TOKEN = re.compile(r"\S+")

# Function words plus question-intent noise ("how long", "what is"). Kept short on purpose:
# aggressive stop lists hurt exact-match queries such as error codes.
STOPWORDS = frozenset(
    ["a", "an", "the", "and", "or", "but", "if", "then", "else", "of", "to", "in", "on", "at", "by", "for", "with", "from", "as", "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "doing", "done", "have", "has", "had", "having", "it", "its", "this", "that", "these", "those", "i", "you", "he", "she", "we", "they", "them", "my", "your", "our", "their", "me", "us", "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "can", "could", "should", "would", "will", "shall", "may", "might", "must", "not", "no", "nor", "so", "than", "too", "very", "just", "also", "into", "onto", "over", "under", "about", "after", "before", "during", "up", "down", "out", "off", "again", "once", "any", "all", "each", "few", "more", "most", "other", "some", "such", "only", "own", "same", "there", "here", "s", "t", "don", "doesn", "didn", "isn", "aren", "wasn", "weren", "won", "wouldn", "couldn", "shouldn", "dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "werent", "wont", "cant", "long", "often", "many", "much", "soon"]
)


def stem(token: str) -> str:
    """Strip common English suffixes. Tokens containing digits are left untouched."""
    if len(token) <= 3 or any(ch.isdigit() for ch in token):
        return token
    t = token
    if t.endswith("ies") and len(t) > 4:
        t = t[:-3] + "y"
    elif t.endswith("sses"):
        t = t[:-2]
    elif t.endswith("s") and not t.endswith(("ss", "us", "is")):
        t = t[:-1]
    for suffix in ("ing", "ed"):
        if t.endswith(suffix) and len(t) - len(suffix) >= 4:
            t = t[: -len(suffix)]
            if len(t) >= 2 and t[-1] == t[-2] and t[-1] not in "lsz":
                t = t[:-1]  # logging -> logg -> log
            break
    if t.endswith("e") and len(t) > 4:
        t = t[:-1]  # rotate -> rotat, so it meets "rotated" -> "rotat"
    return t


def tokenize(text: str, *, stem_tokens: bool = True, drop_stopwords: bool = True) -> list[str]:
    """Lower-case, strip accents' compatibility forms, split on non-alphanumerics."""
    normalised = unicodedata.normalize("NFKC", text).lower().replace("'", "").replace("’", "")
    out: list[str] = []
    for match in _WORD.finditer(normalised):
        word = match.group()
        if drop_stopwords and word in STOPWORDS:
            continue
        out.append(stem(word) if stem_tokens else word)
    return out


def count_tokens(text: str) -> int:
    """Cheap whitespace token count used for chunk sizing (a proxy for model tokens)."""
    return len(_WHITESPACE_TOKEN.findall(text))
