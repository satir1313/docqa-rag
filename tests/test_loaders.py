import pytest

from docqa.ingestion.loaders import load_documents, parse_front_matter, slugify


def minimal_pdf(text: str) -> bytes:
    content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n"
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


def test_front_matter_parsing_splits_tags_and_strips_block():
    meta, body = parse_front_matter("---\ntitle: T\ntags: a, b ,c\n---\n# Body\n")
    assert meta == {"title": "T", "tags": ["a", "b", "c"]}
    assert body.strip() == "# Body"


def test_unterminated_front_matter_is_treated_as_body():
    raw = "---\ntitle: T\nno closing"
    assert parse_front_matter(raw) == ({}, raw)


def test_load_documents_reads_md_and_txt_with_metadata(tmp_path):
    (tmp_path / "Alpha Guide.md").write_text("---\ntitle: Alpha\ntags: x\n---\n# Alpha\nbody")
    (tmp_path / "notes.txt").write_text("plain text file")
    docs, skipped = load_documents(tmp_path)
    by_id = {d.doc_id: d for d in docs}
    assert set(by_id) == {"alpha-guide", "notes"}
    assert by_id["alpha-guide"].title == "Alpha"
    assert by_id["alpha-guide"].metadata["tags"] == ["x"]
    assert by_id["notes"].title == "notes"
    assert skipped == []


def test_bad_files_are_skipped_and_reported_not_fatal(tmp_path):
    (tmp_path / "ok.md").write_text("# ok\ncontent")
    (tmp_path / "empty.md").write_text("   \n")
    (tmp_path / "binary.txt").write_bytes(b"\xff\xfe\x00\x80\x81")
    (tmp_path / "ignored.png").write_bytes(b"not a doc")
    docs, skipped = load_documents(tmp_path)
    assert [d.doc_id for d in docs] == ["ok"]
    assert len(skipped) == 2
    assert any("empty.md" in s for s in skipped)


def test_duplicate_doc_ids_fail_loudly(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "same.md").write_text("one")
    (tmp_path / "b" / "same.md").write_text("two")
    with pytest.raises(ValueError, match="duplicate doc_id"):
        load_documents(tmp_path)


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "nope")


def test_loading_is_deterministic_and_hash_changes_with_content(tmp_path):
    (tmp_path / "a.md").write_text("one")
    first, _ = load_documents(tmp_path)
    (tmp_path / "a.md").write_text("two")
    second, _ = load_documents(tmp_path)
    assert first[0].content_hash != second[0].content_hash


def test_pdf_text_is_extracted(tmp_path):
    pytest.importorskip("pypdf")
    (tmp_path / "doc.pdf").write_bytes(minimal_pdf("Hello retrieval world"))
    docs, skipped = load_documents(tmp_path)
    assert skipped == []
    assert "Hello retrieval world" in docs[0].text


def test_slugify():
    assert slugify("Data Retention (v2)!") == "data-retention-v2"
    assert slugify("!!!") == "doc"
