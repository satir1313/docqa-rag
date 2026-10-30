from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from docqa.config import Settings
from docqa.pipeline import RagPipeline, build_index

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"
GOLDEN = ROOT / "data" / "eval" / "golden.jsonl"


@pytest.fixture(scope="session")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    return replace(
        Settings(), corpus_dir=CORPUS, index_dir=tmp_path_factory.mktemp("index") / "idx"
    )


@pytest.fixture(scope="session")
def pipeline(settings: Settings) -> RagPipeline:
    """A real index built from the bundled sample corpus, shared across tests."""
    build_index(settings)
    return RagPipeline.load(settings)
