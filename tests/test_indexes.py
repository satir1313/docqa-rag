import numpy as np

from docqa.indexing import BM25Index, VectorIndex


def test_bm25_rare_term_outranks_common_term():
    docs = [["common", "rare"], ["common", "filler"], ["common", "other"], ["common", "x"]]
    index = BM25Index(docs)
    ranked = index.search(["rare", "common"], k=4)
    assert ranked[0][0] == 0


def test_bm25_prefers_higher_term_frequency_and_shorter_docs():
    index = BM25Index([["cat", "cat", "cat", "dog"], ["cat", "dog", "bird", "fish"], ["x"]])
    ranked = [i for i, _ in index.search(["cat"], k=3)]
    assert ranked == [0, 1]  # doc 2 has no match and is not returned


def test_bm25_respects_allowed_mask_and_unknown_terms():
    index = BM25Index([["a"], ["a"], ["b"]])
    allowed = np.array([False, True, True])
    assert [i for i, _ in index.search(["a"], k=5, allowed=allowed)] == [1]
    assert index.search(["zzz"], k=5) == []
    assert index.search([], k=5) == []


def test_bm25_is_deterministic_on_ties():
    index = BM25Index([["a"], ["a"], ["a"]])
    assert [i for i, _ in index.search(["a"], k=3)] == [0, 1, 2]


def test_bm25_handles_empty_corpus():
    assert BM25Index([]).search(["a"], k=3) == []


def test_vector_search_orders_by_cosine_and_filters_non_positive():
    matrix = np.array([[1, 0], [0.8, 0.6], [0, 1], [-1, 0]], dtype=np.float32)
    ranked = VectorIndex(matrix).search(np.array([1, 0], dtype=np.float32), k=4)
    assert [i for i, _ in ranked] == [0, 1]  # orthogonal and negative are dropped


def test_vector_search_respects_mask_and_k():
    matrix = np.eye(3, dtype=np.float32)
    index = VectorIndex(matrix)
    q = np.array([1, 1, 0], dtype=np.float32)
    assert {i for i, _ in index.search(q, k=1)} <= {0, 1}
    assert [i for i, _ in index.search(q, k=3, allowed=np.array([False, True, True]))] == [1]
    assert index.search(q, k=0) == []
