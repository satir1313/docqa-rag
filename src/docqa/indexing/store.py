"""On-disk index format with integrity checks and atomic replacement.

Layout of an index directory:
    manifest.json   schema version, embedder identity, counts, per-document hashes, timestamps
    chunks.jsonl    one Chunk per line
    vectors.npy     float32 (n_chunks, dim) matrix aligned with chunks.jsonl
    lsa_*.{npz,json} embedder state (embedder specific)

`save` writes into a sibling temp directory and swaps it in, so a crash mid-build can never
leave a half-written index at the serving path.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from docqa.embeddings import Embedder, load_embedder
from docqa.schemas import Chunk, Document

SCHEMA_VERSION = 1


class IndexError_(RuntimeError):  # noqa: N801 - avoid shadowing the builtin IndexError
    """Raised when an index on disk is missing, corrupt or incompatible."""


@dataclass
class LoadedIndex:
    chunks: list[Chunk]
    vectors: np.ndarray
    embedder: Embedder
    manifest: dict[str, Any]


class IndexStore:
    @staticmethod
    def save(
        directory: Path,
        *,
        chunks: list[Chunk],
        vectors: np.ndarray,
        embedder: Embedder,
        documents: list[Document],
        settings_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        if len(chunks) != vectors.shape[0]:
            raise ValueError("chunks and vectors are misaligned")
        directory = Path(directory)
        tmp = directory.with_name(directory.name + ".tmp")
        backup = directory.with_name(directory.name + ".old")
        for leftover in (tmp, backup):
            shutil.rmtree(leftover, ignore_errors=True)
        tmp.mkdir(parents=True)

        embedder_params = embedder.save(tmp)
        np.save(tmp / "vectors.npy", vectors.astype(np.float32), allow_pickle=False)
        with (tmp / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(chunk.model_dump_json() + "\n")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "embedder": {"name": embedder.name, "params": embedder_params},
            "num_chunks": len(chunks),
            "num_documents": len(documents),
            "dim": int(vectors.shape[1]),
            "documents": {d.doc_id: {"hash": d.content_hash, "path": d.source_path} for d in documents},
            "settings": settings_snapshot,
        }
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        if directory.exists():
            directory.rename(backup)
        tmp.rename(directory)
        shutil.rmtree(backup, ignore_errors=True)
        return manifest

    @staticmethod
    def load(directory: Path) -> LoadedIndex:
        directory = Path(directory)
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"no index at {directory} (run `docqa ingest` first)")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise IndexError_(
                f"index schema {manifest.get('schema_version')} != supported {SCHEMA_VERSION}; "
                "rebuild with `docqa ingest`"
            )
        chunks = [
            Chunk.model_validate_json(line)
            for line in (directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        vectors = np.load(directory / "vectors.npy", allow_pickle=False)
        if len(chunks) != vectors.shape[0] or len(chunks) != manifest["num_chunks"]:
            raise IndexError_("index is corrupt: chunk/vector/manifest counts disagree")
        spec = manifest["embedder"]
        embedder = load_embedder(spec["name"], spec["params"], directory)
        return LoadedIndex(chunks=chunks, vectors=vectors, embedder=embedder, manifest=manifest)
