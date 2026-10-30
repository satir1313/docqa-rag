"""Structure-aware chunking for Markdown-like text.

Why not fixed-size windows? They cut tables in half, separate a heading from its content and
fragment code fences, which is the most common silent cause of poor retrieval. This chunker:

1. parses the document into blocks (heading, paragraph, list, table, code),
2. tracks the heading path ("Policy > Credential rotation"), attached to every chunk,
3. never mixes content from two sections into one chunk,
4. packs whole blocks up to `max_tokens`, carrying whole trailing blocks as overlap,
5. only splits a block (by line, sentence, then word) when it alone exceeds `max_tokens`.

Hard guarantee (tested): no chunk exceeds `max_tokens`.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from docqa.schemas import Chunk, Document
from docqa.text import count_tokens

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^(```|~~~)")
_LIST = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class ChunkerConfig:
    max_tokens: int = 120
    overlap_tokens: int = 25

    def __post_init__(self) -> None:
        if self.max_tokens < 10:
            raise ValueError("max_tokens must be >= 10")
        if not 0 <= self.overlap_tokens < self.max_tokens:
            raise ValueError("overlap_tokens must satisfy 0 <= overlap < max_tokens")


@dataclass(frozen=True)
class Block:
    kind: str  # heading | paragraph | list | table | code
    text: str
    level: int = 0


def parse_blocks(text: str) -> list[Block]:
    blocks: list[Block] = []
    buf: list[str] = []
    buf_kind: str | None = None
    fence: str | None = None

    def flush() -> None:
        nonlocal buf, buf_kind
        if buf:
            blocks.append(Block(buf_kind or "paragraph", "\n".join(buf)))
        buf, buf_kind = [], None

    for line in text.splitlines():
        stripped = line.strip()
        if fence:
            buf.append(line)
            if stripped.startswith(fence):
                flush()
                fence = None
            continue
        opening = _FENCE.match(stripped)
        if opening:
            flush()
            fence, buf, buf_kind = opening.group(1), [line], "code"
            continue
        if not stripped:
            flush()
            continue
        heading = _HEADING.match(line)
        if heading:
            flush()
            blocks.append(Block("heading", heading.group(2).strip(), len(heading.group(1))))
            continue
        if _LIST.match(line):
            kind = "list"
        elif stripped.startswith("|"):
            kind = "table"
        elif buf_kind == "list" and line[:1] in {" ", "\t"}:
            kind = "list"  # indented continuation of a list item
        else:
            kind = "paragraph"
        if buf_kind and kind != buf_kind:
            flush()
        buf_kind = kind
        buf.append(line)
    flush()
    return blocks


def _hard_split(piece: str, max_tokens: int) -> list[str]:
    words = piece.split()
    return [" ".join(words[i : i + max_tokens]) for i in range(0, len(words), max_tokens)]


def _split_oversized(block: Block, max_tokens: int) -> list[Block]:
    """Split a block that is larger than `max_tokens`: by line, then sentence, then word."""
    if count_tokens(block.text) <= max_tokens:
        return [block]
    line_based = block.kind in {"list", "table", "code"}
    pieces = block.text.split("\n") if line_based else _SENTENCE_BOUNDARY.split(block.text)
    joiner = "\n" if line_based else " "
    flat: list[str] = []
    for piece in pieces:
        flat.extend([piece] if count_tokens(piece) <= max_tokens else _hard_split(piece, max_tokens))
    out: list[Block] = []
    current: list[str] = []
    size = 0
    for piece in flat:
        n = count_tokens(piece)
        if current and size + n > max_tokens:
            out.append(Block(block.kind, joiner.join(current)))
            current, size = [], 0
        current.append(piece)
        size += n
    if current:
        out.append(Block(block.kind, joiner.join(current)))
    return out


def _pack(blocks: list[Block], cfg: ChunkerConfig) -> list[str]:
    """Pack blocks of ONE section into chunk texts, with whole-block overlap."""
    pieces = [p for b in blocks for p in _split_oversized(b, cfg.max_tokens)]
    chunks: list[str] = []
    current: list[Block] = []
    size = 0
    fresh = 0  # blocks in `current` that are not already part of the previous chunk
    for piece in pieces:
        n = count_tokens(piece.text)
        if current and size + n > cfg.max_tokens:
            if fresh:
                chunks.append("\n\n".join(b.text for b in current))
            carry: list[Block] = []
            carried = 0
            for prev in reversed(current):
                t = count_tokens(prev.text)
                if carried + t > cfg.overlap_tokens:
                    break
                carry.insert(0, prev)
                carried += t
            while carry and carried + n > cfg.max_tokens:
                carried -= count_tokens(carry.pop(0).text)
            current, size, fresh = carry, carried, 0
        current.append(piece)
        size += n
        fresh += 1
    if fresh:
        chunks.append("\n\n".join(b.text for b in current))
    return chunks


def _sections(blocks: list[Block], fallback_title: str) -> Iterator[tuple[list[str], list[Block]]]:
    stack: list[tuple[int, str]] = []
    body: list[Block] = []
    path: list[str] = [fallback_title]

    for block in blocks:
        if block.kind == "heading":
            if body:
                yield path, body
                body = []
            while stack and stack[-1][0] >= block.level:
                stack.pop()
            stack.append((block.level, block.text))
            path = [text for _, text in stack]
        else:
            body.append(block)
    if body:
        yield path, body


def chunk_document(doc: Document, cfg: ChunkerConfig | None = None) -> list[Chunk]:
    cfg = cfg or ChunkerConfig()
    base_meta: dict[str, Any] = {
        **doc.metadata,
        "title": doc.title,
        "source_path": doc.source_path,
    }
    chunks: list[Chunk] = []
    for heading_path, blocks in _sections(parse_blocks(doc.text), doc.title):
        for text in _pack(blocks, cfg):
            ordinal = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::{ordinal:03d}",
                    doc_id=doc.doc_id,
                    ordinal=ordinal,
                    text=text,
                    heading_path=list(heading_path),
                    token_count=count_tokens(text),
                    metadata=dict(base_meta),
                    content_hash=hashlib.sha1(text.encode("utf-8")).hexdigest(),  # noqa: S324
                )
            )
    return chunks
