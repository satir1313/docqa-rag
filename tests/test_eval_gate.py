"""Regression gate: thresholds sit well below observed values, so only real regressions trip."""
import pytest

from docqa.evaluation import evaluate, load_golden
from tests.conftest import GOLDEN

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def report(pipeline):
    return evaluate(pipeline, load_golden(GOLDEN))


def test_golden_set_shape():
    items = load_golden(GOLDEN)
    assert len(items) >= 40
    assert {i.split for i in items} == {"dev", "test"}
    assert any(not i.answerable for i in items)


def test_default_mode_quality_floor(report):
    m = report["retrieval"]["hybrid"]
    assert m["recall@5"] >= 0.85 and m["mrr@10"] >= 0.75


def test_hybrid_not_worse_than_bm25_by_more_than_noise(report):
    assert report["retrieval"]["hybrid"]["mrr@10"] >= report["retrieval"]["bm25"]["mrr@10"] - 0.03


def test_report_has_expected_sections(report):
    for key in ("retrieval_by_category", "retrieval_by_split", "answers", "threshold_sweep"):
        assert key in report
