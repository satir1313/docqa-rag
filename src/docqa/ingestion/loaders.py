"""Document loading: Markdown / plain text (with optional front matter) and PDF.

Failure policy: a single unreadable file never aborts ingestion. It is skipped and reported
so operators can see exactly what did not make it into the index.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

from docqa.schemas import Document

log = logging.getLogger("docqa.ingestion")

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}
MAX_FILE_BYTES = 10 * 1024 * 1024
_H1 = re.compile(r"^#\s+(.+?)\s*#*\s*$", re.MULTILINE)


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "doc"


def parse_front_matter(raw: str) -> tuple[dict[str, Any], str]:
    """Parse a minimal `key: value` front-matter block (no YAML dependency).

    `tags` is split on commas into a list. An unterminated block is treated as body text.
    """
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw
    for end in range(1, len(lines)):
        if lines[end].strip() == "---":
            meta: dict[str, Any] = {}
            for line in lines[1:end]:
                if ":" not in line or line.lstrip().startswith("#"):
                    continue
                key, value = line.split(":", 1)
                key, value = key.strip().lower(), value.strip()
                meta[key] = (
                    [v.strip() for v in value.split(",") if v.strip()]
                    if key in {"tags", "audience"}
                    else value
                )
            return meta, "\n".join(lines[end + 1 :]).lstrip("\n")
    return {}, raw


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError("PDF support needs `pip install docqa-rag[pdf]`") from exc
    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n\n".join(p for p in pages if p)


def _build_document(path: Path, root: Path) -> Document:
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"file larger than {MAX_FILE_BYTES} bytes")
    raw = _read_pdf(path) if path.suffix.lower() == ".pdf" else path.read_text(encoding="utf-8")
    meta, body = parse_front_matter(raw)
    body = body.replace("\r\n", "\n").strip()
    if not body:
        raise ValueError("document is empty")
    h1 = _H1.search(body)
    title = str(meta.pop("title", "") or (h1.group(1) if h1 else path.stem))
    relative = path.relative_to(root).as_posix() if path.is_relative_to(root) else path.name
    return Document(
        doc_id=slugify(path.stem),
        source_path=relative,
        title=title,
        text=body,
        metadata=meta,
        content_hash=hashlib.sha256(body.encode("utf-8")).hexdigest(),
    )


def load_documents(root: Path) -> tuple[list[Document], list[str]]:
    """Load every supported file under `root` (sorted, so builds are deterministic).

    Returns (documents, skipped) where `skipped` holds "path: reason" strings.
    Raises ValueError on duplicate doc ids, because silently overwriting is worse than failing.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {root}")
    docs: list[Document] = []
    skipped: list[str] = []
    seen: dict[str, str] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            doc = _build_document(path, root)
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort the run
            reason = f"{path.name}: {exc}"
            log.warning("skipping document", extra={"path": str(path), "reason": str(exc)})
            skipped.append(reason)
            continue
        if doc.doc_id in seen:
            raise ValueError(f"duplicate doc_id {doc.doc_id!r}: {seen[doc.doc_id]} and {path}")
        seen[doc.doc_id] = str(path)
        docs.append(doc)
    return docs, skipped
