import json

import numpy as np
import pytest

from docqa.embeddings import LSAEmbedder, make_embedder
from docqa.indexing import IndexStore
from docqa.indexing.store import IndexError_
from docqa.ingestion import chunk_document
from docqa.schemas import Document

TEXTS = [
    "rotate api tokens every ninety days",
    "service account credentials rotate automatically",
    "backups are retained for thirty five days",
    "daily backups stored in a separate region",
    "canary deploys receive five percent of traffic",
    "rollback restores the previous artifact quickly",
]


def fitted() -> LSAEmbedder:
    embedder = LSAEmbedder(dim=4)
    embedder.fit(TEXTS)
    return embedder


def test_vectors_are_unit_norm_and_correct_shape():
    e = fitted()
    vectors = e.embed_documents(TEXTS)
    assert vectors.shape == (len(TEXTS), e.dim)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_related_texts_are_closer_than_unrelated():
    e = fitted()
    q = e.embed_query("how often are tokens rotated")
    sims = e.embed_documents(TEXTS) @ q
    assert sims[0] > sims[2] and sims[0] > sims[4]


def test_unseen_query_gives_zero_vector_not_nan():
    q = fitted().embed_query("zzzz qqqq")
    assert np.isfinite(q).all()
    assert np.linalg.norm(q) == 0.0


def test_save_load_roundtrip_is_exact(tmp_path):
    e = fitted()
    params = e.save(tmp_path)
    loaded = LSAEmbedder.load(tmp_path, params)
    assert loaded.dim == e.dim
    assert np.array_equal(e.embed_documents(TEXTS), loaded.embed_documents(TEXTS))


def test_state_is_saved_without_pickle(tmp_path):
    fitted().save(tmp_path)
    with np.load(tmp_path / "lsa_state.npz", allow_pickle=False) as state:
        assert set(state.files) == {"idf", "components"}
    json.loads((tmp_path / "lsa_vocab.json").read_text())


def test_unfitted_and_tiny_inputs_are_rejected():
    with pytest.raises(RuntimeError):
        LSAEmbedder().embed_query("x")
    with pytest.raises(ValueError):
        LSAEmbedder().fit(["only one"])
    with pytest.raises(ValueError):
        LSAEmbedder(dim=1)


def test_factory_rejects_unknown_embedder():
    with pytest.raises(ValueError):
        make_embedder("nope")


def build_store_inputs():
    doc = Document(
        doc_id="d",
        source_path="d.md",
        title="D",
        text="x",
        content_hash="h",
    )
    chunks = []
    for i, text in enumerate(TEXTS):
        chunks += chunk_document(doc.model_copy(update={"doc_id": f"d{i}", "text": f"# H\n\n{text}"}))
    embedder = fitted()
    return doc, chunks, embedder, embedder.embed_documents([c.index_text for c in chunks])


def test_store_roundtrip_and_no_leftover_temp_dirs(tmp_path):
    doc, chunks, embedder, vectors = build_store_inputs()
    target = tmp_path / "idx"
    manifest = IndexStore.save(
        target, chunks=chunks, vectors=vectors, embedder=embedder, documents=[doc],
        settings_snapshot={"k": 1},
    )
    loaded = IndexStore.load(target)
    assert [c.chunk_id for c in loaded.chunks] == [c.chunk_id for c in chunks]
    assert np.array_equal(loaded.vectors, vectors)
    assert loaded.manifest["num_chunks"] == manifest["num_chunks"] == len(chunks)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["idx"]


def test_rebuild_replaces_existing_index_atomically(tmp_path):
    doc, chunks, embedder, vectors = build_store_inputs()
    target = tmp_path / "idx"
    for _ in range(2):
        IndexStore.save(
            target, chunks=chunks, vectors=vectors, embedder=embedder, documents=[doc],
            settings_snapshot={},
        )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["idx"]


def test_missing_index_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="docqa ingest"):
        IndexStore.load(tmp_path / "none")


def test_schema_version_mismatch_is_rejected(tmp_path):
    doc, chunks, embedder, vectors = build_store_inputs()
    target = tmp_path / "idx"
    IndexStore.save(
        target, chunks=chunks, vectors=vectors, embedder=embedder, documents=[doc],
        settings_snapshot={},
    )
    manifest = json.loads((target / "manifest.json").read_text())
    manifest["schema_version"] = 999
    (target / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(IndexError_, match="schema"):
        IndexStore.load(target)


def test_corrupt_index_is_detected(tmp_path):
    doc, chunks, embedder, vectors = build_store_inputs()
    target = tmp_path / "idx"
    IndexStore.save(
        target, chunks=chunks, vectors=vectors, embedder=embedder, documents=[doc],
        settings_snapshot={},
    )
    lines = (target / "chunks.jsonl").read_text().splitlines()
    (target / "chunks.jsonl").write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(IndexError_, match="corrupt"):
        IndexStore.load(target)


def test_misaligned_save_is_refused(tmp_path):
    doc, chunks, embedder, vectors = build_store_inputs()
    with pytest.raises(ValueError):
        IndexStore.save(
            tmp_path / "idx", chunks=chunks[:-1], vectors=vectors, embedder=embedder,
            documents=[doc], settings_snapshot={},
        )
