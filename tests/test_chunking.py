import pytest

from docqa.ingestion import ChunkerConfig, chunk_document, load_documents
from docqa.ingestion.chunking import parse_blocks
from docqa.schemas import Document
from tests.conftest import CORPUS


def make_doc(text: str, doc_id: str = "d", title: str = "Doc", metadata=None) -> Document:
    return Document(
        doc_id=doc_id,
        source_path=f"{doc_id}.md",
        title=title,
        text=text,
        metadata=metadata or {},
        content_hash="x",
    )


def words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_heading_path_tracks_nesting_and_resets_on_sibling():
    doc = make_doc("# A\n\n## B\ntext b\n\n### C\ntext c\n\n## D\ntext d\n")
    paths = {c.text: c.heading_path for c in chunk_document(doc)}
    assert paths["text b"] == ["A", "B"]
    assert paths["text c"] == ["A", "B", "C"]
    assert paths["text d"] == ["A", "D"]


def test_content_before_first_heading_uses_document_title():
    chunks = chunk_document(make_doc("intro text\n\n# H\nbody", title="My Title"))
    assert chunks[0].heading_path == ["My Title"]


@pytest.mark.parametrize("max_tokens,overlap", [(10, 0), (20, 5), (40, 10), (120, 25)])
def test_no_chunk_ever_exceeds_max_tokens_on_real_corpus(max_tokens, overlap):
    cfg = ChunkerConfig(max_tokens=max_tokens, overlap_tokens=overlap)
    docs, _ = load_documents(CORPUS)
    chunks = [c for d in docs for c in chunk_document(d, cfg)]
    assert chunks
    assert max(c.token_count for c in chunks) <= max_tokens


def test_oversized_paragraph_is_split_but_stays_within_limit():
    doc = make_doc("# T\n\n" + ". ".join(words(8, f"s{i}x") for i in range(10)) + ".")
    chunks = chunk_document(doc, ChunkerConfig(max_tokens=20, overlap_tokens=0))
    assert len(chunks) > 1
    assert all(c.token_count <= 20 for c in chunks)


def test_single_giant_token_run_is_hard_split():
    chunks = chunk_document(make_doc("# T\n\n" + words(95)), ChunkerConfig(max_tokens=30, overlap_tokens=0))
    assert all(c.token_count <= 30 for c in chunks)
    assert sum(c.token_count for c in chunks) == 95


def test_chunks_never_mix_sections():
    doc = make_doc("# T\n\n## One\nalpha beta\n\n## Two\ngamma delta\n")
    for chunk in chunk_document(doc):
        assert not ("alpha" in chunk.text and "gamma" in chunk.text)


def test_code_fence_is_one_block_even_with_blank_lines_inside():
    text = "para\n\n```python\nx = 1\n\ny = 2\n```\n\nafter"
    kinds = [(b.kind, b.text) for b in parse_blocks(text)]
    assert ("code", "```python\nx = 1\n\ny = 2\n```") in kinds


def test_heading_inside_code_fence_is_not_a_heading():
    blocks = parse_blocks("```\n# not a heading\n```")
    assert [b.kind for b in blocks] == ["code"]


def test_table_rows_stay_together():
    doc = make_doc("# T\n\n| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n")
    chunks = chunk_document(doc, ChunkerConfig(max_tokens=50, overlap_tokens=0))
    assert len(chunks) == 1
    assert "| 3 | 4 |" in chunks[0].text


def test_list_items_are_one_block_with_continuations():
    blocks = parse_blocks("- one\n  continued\n- two")
    assert [b.kind for b in blocks] == ["list"]


def test_overlap_repeats_trailing_block_in_next_chunk():
    paras = [words(10, f"p{i}_") for i in range(4)]
    doc = make_doc("# T\n\n" + "\n\n".join(paras))
    chunks = chunk_document(doc, ChunkerConfig(max_tokens=25, overlap_tokens=10))
    assert len(chunks) >= 2
    assert paras[1] in chunks[0].text and paras[1] in chunks[1].text


def test_no_overlap_means_no_duplicated_blocks():
    paras = [words(10, f"p{i}_") for i in range(4)]
    chunks = chunk_document(
        make_doc("# T\n\n" + "\n\n".join(paras)), ChunkerConfig(max_tokens=25, overlap_tokens=0)
    )
    joined = "\n".join(c.text for c in chunks)
    assert all(joined.count(p) == 1 for p in paras)


def test_chunk_ids_are_stable_ordered_and_unique():
    doc = make_doc("# T\n\n## A\none\n\n## B\ntwo\n")
    first = chunk_document(doc)
    second = chunk_document(doc)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second] == ["d::000", "d::001"]


def test_document_metadata_is_propagated_to_chunks():
    doc = make_doc("# T\n\nbody", metadata={"tags": ["x", "y"], "audience": ["sre"]})
    chunk = chunk_document(doc)[0]
    assert chunk.metadata["tags"] == ["x", "y"]
    assert chunk.metadata["title"] == "Doc"


def test_index_text_prepends_section_as_context():
    chunk = chunk_document(make_doc("# Policy\n\n## Retention\n30 days"))[0]
    assert chunk.index_text.startswith("Policy > Retention\n")


@pytest.mark.parametrize("kwargs", [{"max_tokens": 5}, {"max_tokens": 50, "overlap_tokens": 50}])
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ChunkerConfig(**kwargs)


def test_empty_sections_produce_no_chunks():
    assert chunk_document(make_doc("# Only a heading\n\n## Another\n")) == []
