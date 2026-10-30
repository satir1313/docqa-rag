"""Evaluation harness: retrieval ablation, answer checks, abstention and a CI quality gate.

Relevance is defined by (doc_id, section heading substring), NOT by chunk id, so the golden set
stays valid when you change chunk size, overlap or the chunker itself.

What is measured
- Retrieval (answerable questions): Recall@{1,3,5} over required references, MRR@10, nDCG@10,
  latency p50/p95, for each retrieval mode (bm25, dense, hybrid, hybrid_rerank).
- Answers (hybrid_rerank + the configured generator): keyword coverage, citation precision,
  abstention on unanswerable questions and false abstention on answerable ones.
- Per-category and per-split breakdowns, plus a support-threshold sweep.

What this does NOT prove: with a small hand-written golden set these are regression signals,
not benchmark results. See docs/EVALUATION.md for the caveats.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from docqa.evaluation.metrics import mean, ndcg_at_k, percentile, reciprocal_rank, ref_recall_at_k
from docqa.generation.base import compute_support
from docqa.pipeline import RagPipeline
from docqa.schemas import Chunk, RetrievalMode

KS = (1, 3, 5)
MODES = (
    RetrievalMode.BM25,
    RetrievalMode.DENSE,
    RetrievalMode.HYBRID,
    RetrievalMode.HYBRID_RERANK,
)


class Reference(BaseModel):
    doc_id: str
    section_contains: str | None = None


class GoldenItem(BaseModel):
    id: str
    question: str
    category: str = "uncategorised"
    split: str = "test"
    relevant: list[Reference] = Field(default_factory=list)
    answerable: bool = True
    expected_keywords: list[str] = Field(default_factory=list)


def load_golden(path: Path) -> list[GoldenItem]:
    items = [
        GoldenItem.model_validate_json(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError("golden set contains duplicate ids")
    for item in items:
        if item.answerable and not item.relevant:
            raise ValueError(f"{item.id}: answerable items need at least one reference")
    return items


def _matches(chunk: Chunk, ref: Reference) -> bool:
    if chunk.doc_id != ref.doc_id:
        return False
    return ref.section_contains is None or ref.section_contains.lower() in chunk.section.lower()


def _score_retrieval(hits: list[Chunk], item: GoldenItem, corpus: list[Chunk]) -> dict[str, float]:
    flags = [any(_matches(c, r) for r in item.relevant) for c in hits]
    best_ranks: list[int | None] = []
    for ref in item.relevant:
        best_ranks.append(next((i for i, c in enumerate(hits, 1) if _matches(c, ref)), None))
    total_relevant = sum(1 for c in corpus if any(_matches(c, r) for r in item.relevant))
    out = {f"recall@{k}": ref_recall_at_k(best_ranks, k) for k in KS}
    out["mrr@10"] = reciprocal_rank(flags, 10)
    out["ndcg@10"] = ndcg_at_k(flags, total_relevant, 10)
    return out


def _summarise(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = [f"recall@{k}" for k in KS] + ["mrr@10", "ndcg@10"]
    return {key: round(mean([r[key] for r in rows]), 4) for key in keys}


def evaluate(
    pipeline: RagPipeline,
    golden: list[GoldenItem],
    *,
    k_answer: int = 5,
    thresholds: tuple[float, ...] = (0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6),
) -> dict[str, Any]:
    corpus = pipeline.retriever.chunks
    answerable = [g for g in golden if g.answerable]

    # ---- retrieval ablation ---------------------------------------------------------------
    retrieval: dict[str, Any] = {}
    per_item: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for mode in MODES:
        pipeline.search(answerable[0].question, k=10, mode=mode)  # warm-up (caches, imports)
        rows, latencies = [], []
        for item in answerable:
            t = time.perf_counter()
            result = pipeline.search(item.question, k=10, mode=mode)
            latencies.append((time.perf_counter() - t) * 1000)
            scores = _score_retrieval([h.chunk for h in result.hits], item, corpus)
            per_item[item.id][mode.value] = scores
            rows.append(scores)
        retrieval[mode.value] = {
            **_summarise(rows),
            "latency_ms_p50": round(percentile(latencies, 50), 3),
            "latency_ms_p95": round(percentile(latencies, 95), 3),
        }

    by_category: dict[str, dict[str, Any]] = {}
    for category in sorted({g.category for g in answerable}):
        by_category[category] = {
            mode.value: _summarise(
                [per_item[g.id][mode.value] for g in answerable if g.category == category]
            )
            | {"n": sum(1 for g in answerable if g.category == category)}
            for mode in MODES
        }
    by_split: dict[str, dict[str, Any]] = {}
    for split in sorted({g.split for g in answerable}):
        by_split[split] = {
            mode.value: _summarise(
                [per_item[g.id][mode.value] for g in answerable if g.split == split]
            )
            | {"n": sum(1 for g in answerable if g.split == split)}
            for mode in MODES
        }

    # ---- answers + abstention -------------------------------------------------------------
    answer_rows: list[dict[str, Any]] = []
    for item in golden:
        answer, result = pipeline.ask(item.question, k=k_answer)
        cited_ok = [
            any(_matches(result.hits[c.index - 1].chunk, r) for r in item.relevant)
            for c in answer.citations
            if 1 <= c.index <= len(result.hits)
        ]
        keyword_hit = bool(item.expected_keywords) and all(
            kw.lower() in answer.text.lower() for kw in item.expected_keywords
        )
        answer_rows.append(
            {
                "id": item.id,
                "split": item.split,
                "answerable": item.answerable,
                "abstained": answer.abstained,
                "support": compute_support(item.question, result.hits),
                "keyword_hit": keyword_hit,
                "citation_precision": mean([float(x) for x in cited_ok]) if cited_ok else None,
            }
        )
    ans = [r for r in answer_rows if r["answerable"]]
    unans = [r for r in answer_rows if not r["answerable"]]
    cite = [r["citation_precision"] for r in ans if r["citation_precision"] is not None]
    answers = {
        "generator": pipeline.generator.name,
        "retrieval_mode": pipeline.settings.default_mode,
        "min_support": pipeline.settings.min_support,
        "answerable": len(ans),
        "unanswerable": len(unans),
        "keyword_coverage": round(mean([float(r["keyword_hit"]) for r in ans]), 4),
        "citation_precision": round(mean(cite), 4),
        "false_abstention_rate": round(mean([float(r["abstained"]) for r in ans]), 4),
        "correct_abstention_rate": round(mean([float(r["abstained"]) for r in unans]), 4),
    }

    def _sweep(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
        pos = [r for r in rows if r["answerable"]]
        neg = [r for r in rows if not r["answerable"]]
        return [
            {
                "threshold": th,
                "false_abstention_rate": round(mean([float(r["support"] < th) for r in pos]), 4),
                "correct_abstention_rate": round(mean([float(r["support"] < th) for r in neg]), 4),
                "n_answerable": len(pos),
                "n_unanswerable": len(neg),
            }
            for th in thresholds
        ]

    sweep = {
        "all": _sweep(answer_rows),
        "dev": _sweep([r for r in answer_rows if r["split"] == "dev"]),
        "test": _sweep([r for r in answer_rows if r["split"] == "test"]),
    }
    # Selection rule fixed in advance and applied to DEV ONLY: the largest threshold whose
    # false-abstention rate stays <= 10%. The test split is then reported, never tuned on.
    eligible = [r for r in sweep["dev"] if r["false_abstention_rate"] <= 0.10]
    recommended = max((r["threshold"] for r in eligible), default=None)

    return {
        "n_questions": len(golden),
        "n_chunks": len(corpus),
        "retrieval": retrieval,
        "retrieval_by_category": by_category,
        "retrieval_by_split": by_split,
        "answers": answers,
        "threshold_sweep": sweep,
        "recommended_threshold_from_dev": recommended,
        "failures": _failures(per_item, answerable, pipeline.settings.default_mode),
    }


def _failures(
    per_item: dict[str, dict[str, dict[str, float]]], items: list[GoldenItem], mode: str
) -> list[dict[str, Any]]:
    """Questions whose required reference was NOT in the top 5, for error analysis."""
    out = []
    for item in items:
        score = per_item[item.id][mode]
        if score["recall@5"] < 1.0:
            out.append(
                {
                    "id": item.id,
                    "question": item.question,
                    "category": item.category,
                    "recall@5": score["recall@5"],
                }
            )
    return out


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Evaluation report",
        "",
        f"{report['n_questions']} golden questions over {report['n_chunks']} chunks.",
        "",
        "## Retrieval ablation (answerable questions)",
        "",
        "| Mode | Recall@1 | Recall@3 | Recall@5 | MRR@10 | nDCG@10 | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for mode, m in report["retrieval"].items():
        lines.append(
            f"| {mode} | {m['recall@1']:.3f} | {m['recall@3']:.3f} | {m['recall@5']:.3f} | "
            f"{m['mrr@10']:.3f} | {m['ndcg@10']:.3f} | {m['latency_ms_p50']:.2f} | "
            f"{m['latency_ms_p95']:.2f} |"
        )
    for title, key in (("category", "retrieval_by_category"), ("split", "retrieval_by_split")):
        lines += ["", f"## Recall@5 by {title}", "", "| " + title + " | n | " + " | ".join(
            report["retrieval"]
        ) + " |", "|---|---|" + "---|" * len(report["retrieval"])]
        for name, modes in report[key].items():
            n = next(iter(modes.values()))["n"]
            lines.append(
                f"| {name} | {n} | "
                + " | ".join(f"{modes[m]['recall@5']:.3f}" for m in report["retrieval"])
                + " |"
            )
    a = report["answers"]
    lines += [
        "",
        f"## Answers (generator: `{a['generator']}`, retrieval: `{a['retrieval_mode']}`, "
        f"min_support={a['min_support']})",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Keyword coverage (answerable, n={a['answerable']}) | {a['keyword_coverage']:.3f} |",
        f"| Citation precision | {a['citation_precision']:.3f} |",
        f"| False abstention rate (answerable) | {a['false_abstention_rate']:.3f} |",
        f"| Correct abstention rate (unanswerable, n={a['unanswerable']}) | "
        f"{a['correct_abstention_rate']:.3f} |",
        "",
        "## Support-threshold sweep",
        "",
    ]
    for split in ("dev", "test", "all"):
        rows = report["threshold_sweep"][split]
        lines += [
            f"**{split}** (answerable n={rows[0]['n_answerable']}, "
            f"unanswerable n={rows[0]['n_unanswerable']})",
            "",
            "| min_support | False abstention (lower is better) | Correct abstention (higher is better) |",
            "|---|---|---|",
        ]
        lines += [
            f"| {row['threshold']} | {row['false_abstention_rate']:.3f} | "
            f"{row['correct_abstention_rate']:.3f} |"
            for row in rows
        ]
        lines.append("")
    lines.append(
        "Threshold chosen on dev only (largest value with <=10% false abstention): "
        f"**{report['recommended_threshold_from_dev']}**."
    )
    lines += ["", f"## Retrieval failures for `{report['answers']['retrieval_mode']}` (required reference not in top 5)", ""]
    if report["failures"]:
        lines += ["| id | category | question | recall@5 |", "|---|---|---|---|"]
        lines += [
            f"| {f['id']} | {f['category']} | {f['question']} | {f['recall@5']:.2f} |"
            for f in report["failures"]
        ]
    else:
        lines.append("None.")
    lines.append("")
    return "\n".join(lines)


def dump_json(report: dict[str, Any], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
