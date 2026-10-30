import math

import pytest

from docqa.evaluation.metrics import mean, ndcg_at_k, percentile, reciprocal_rank, ref_recall_at_k
from docqa.retrieval.filters import matches_filters
from docqa.retrieval.hybrid import reciprocal_rank_fusion
from docqa.schemas import RetrievalMode


def test_rrf_scores_and_ranks_are_hand_computable():
    fused = reciprocal_rank_fusion({"a": [10, 20], "b": [20, 30]}, k=60)
    assert fused[10][0] == pytest.approx(1 / 61)
    assert fused[20][0] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[20][1] == {"a": 2, "b": 1}
    assert fused[30][1] == {"b": 2}


def test_rrf_weights_scale_contribution():
    fused = reciprocal_rank_fusion({"a": [1], "b": [2]}, k=60, weights={"a": 2.0})
    assert fused[1][0] == pytest.approx(2 / 61)
    assert fused[2][0] == pytest.approx(1 / 61)


def test_filters_match_scalars_and_lists():
    meta = {"tags": ["security", "ops"], "audience": "sre"}
    assert matches_filters(meta, None)
    assert matches_filters(meta, {"tags": "security"})
    assert matches_filters(meta, {"tags": ["x", "ops"]})
    assert not matches_filters(meta, {"tags": "billing"})
    assert not matches_filters(meta, {"missing": "x"})
    assert not matches_filters(meta, {"audience": "sre", "tags": "billing"})


def test_reciprocal_rank():
    assert reciprocal_rank([False, True, True], 10) == 0.5
    assert reciprocal_rank([False, False, True], 2) == 0.0
    assert reciprocal_rank([], 5) == 0.0


def test_ndcg_matches_hand_calculation():
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k([True, False, True], 2, 3) == pytest.approx(dcg / ideal)
    assert ndcg_at_k([True], 1, 5) == 1.0
    assert ndcg_at_k([False], 0, 5) == 0.0


def test_ref_recall():
    assert ref_recall_at_k([1, 4, None], 3) == pytest.approx(1 / 3)
    assert ref_recall_at_k([1, 4, None], 5) == pytest.approx(2 / 3)
    assert ref_recall_at_k([], 5) == 0.0


def test_percentile_and_mean():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 95) == 5
    assert percentile([], 50) == 0.0
    assert mean([1, 2, 3]) == 2 and mean([]) == 0.0


@pytest.mark.parametrize("mode", list(RetrievalMode))
def test_every_mode_finds_the_obvious_answer(pipeline, mode):
    result = pipeline.search("How long are database backups retained?", k=5, mode=mode)
    assert result.hits
    assert result.hits[0].chunk.doc_id in {"data-retention", "disaster-recovery"}
    assert [h.rank for h in result.hits] == list(range(1, len(result.hits) + 1))


def test_filters_restrict_results(pipeline):
    any_doc = pipeline.search("retention", k=10).hits[0].chunk.metadata
    tag = any_doc["tags"][0] if isinstance(any_doc["tags"], list) else any_doc["tags"]
    hits = pipeline.search("retention", k=10, filters={"tags": tag}).hits
    assert hits
    for h in hits:
        t = h.chunk.metadata["tags"]
        assert tag in (t if isinstance(t, list) else [t])


def test_diversity_cap_limits_chunks_per_document(pipeline):
    hits = pipeline.search("policy", k=10).hits
    per_doc = {}
    for h in hits:
        per_doc[h.chunk.doc_id] = per_doc.get(h.chunk.doc_id, 0) + 1
    assert max(per_doc.values()) <= pipeline.settings.max_per_doc


def test_empty_and_stopword_only_queries_return_nothing(pipeline):
    assert pipeline.search("the of and", k=5).hits == []
