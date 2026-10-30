"""Command line interface: ingest | search | ask | eval | serve."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from docqa.config import Settings
from docqa.evaluation.harness import dump_json, evaluate, load_golden, render_markdown
from docqa.logging_utils import configure_logging
from docqa.pipeline import RagPipeline, build_index
from docqa.schemas import RetrievalMode


def _parse_filters(pairs: list[str] | None) -> dict[str, str | list[str]] | None:
    if not pairs:
        return None
    out: dict[str, str | list[str]] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise SystemExit(f"invalid --filter {pair!r}, expected key=value")
        out[key] = [v for v in value.split(",") if v]
    return out


def _cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    report = build_index(settings, args.corpus, args.index)
    print(json.dumps(report.model_dump(), indent=2))
    return 0


def _cmd_search(args: argparse.Namespace, settings: Settings) -> int:
    pipeline = RagPipeline.load(settings, args.index)
    result = pipeline.search(
        args.query,
        k=args.k,
        filters=_parse_filters(args.filter),
        mode=RetrievalMode(args.mode) if args.mode else None,
    )
    for hit in result.hits:
        print(f"{hit.rank:>2}. [{hit.score:.4f}] {hit.chunk.doc_id} > {hit.chunk.section}")
        print(f"    {hit.chunk.text[:160].replace(chr(10), ' ')}")
    print(f"\n{json.dumps(result.timings_ms)} ms, {result.candidates} candidates", file=sys.stderr)
    return 0


def _cmd_ask(args: argparse.Namespace, settings: Settings) -> int:
    pipeline = RagPipeline.load(settings, args.index)
    answer, _ = pipeline.ask(args.question, k=args.k, filters=_parse_filters(args.filter))
    print(answer.text)
    for c in answer.citations:
        print(f"  [{c.index}] {c.doc_id} > {c.section}")
    return 0


def _cmd_eval(args: argparse.Namespace, settings: Settings) -> int:
    pipeline = RagPipeline.load(settings, args.index)
    report = evaluate(pipeline, load_golden(args.golden))
    markdown = render_markdown(report)
    print(markdown)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(markdown, encoding="utf-8")
    if args.json:
        dump_json(report, args.json)

    best = report["retrieval"][pipeline.settings.default_mode]
    failures = []
    if args.min_recall_at_5 is not None and best["recall@5"] < args.min_recall_at_5:
        failures.append(f"recall@5 {best['recall@5']:.3f} < {args.min_recall_at_5}")
    if args.min_mrr is not None and best["mrr@10"] < args.min_mrr:
        failures.append(f"mrr@10 {best['mrr@10']:.3f} < {args.min_mrr}")
    if failures:
        print("QUALITY GATE FAILED: " + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


def _cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    import uvicorn

    from docqa.api import create_app

    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_config=None)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="docqa", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="build the index from a corpus directory")
    p.add_argument("--corpus", type=Path, default=None)
    p.add_argument("--index", type=Path, default=None)
    p.set_defaults(func=_cmd_ingest)

    p = sub.add_parser("search", help="retrieve chunks for a query")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--mode", choices=[m.value for m in RetrievalMode], default=None)
    p.add_argument("--filter", action="append", help="metadata filter key=v1,v2 (repeatable)")
    p.add_argument("--index", type=Path, default=None)
    p.set_defaults(func=_cmd_search)

    p = sub.add_parser("ask", help="answer a question with citations")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--filter", action="append")
    p.add_argument("--index", type=Path, default=None)
    p.set_defaults(func=_cmd_ask)

    p = sub.add_parser("eval", help="run the golden-set evaluation (optionally as a CI gate)")
    p.add_argument("--golden", type=Path, default=Path("data/eval/golden.jsonl"))
    p.add_argument("--index", type=Path, default=None)
    p.add_argument("--out", type=Path, help="write the markdown report here")
    p.add_argument("--json", type=Path, help="write the raw JSON report here")
    p.add_argument("--min-recall-at-5", type=float, default=None)
    p.add_argument("--min-mrr", type=float, default=None)
    p.set_defaults(func=_cmd_eval)

    p = sub.add_parser("serve", help="run the HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=_cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    try:
        return int(args.func(args, settings))
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
