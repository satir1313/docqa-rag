.PHONY: install test lint ingest eval serve docker
install:
	pip install -e ".[dev]"
lint:
	ruff check .
test:
	pytest -q
ingest:
	docqa ingest
eval: ingest
	docqa eval --out docs/eval-report.md --json var/eval.json --min-recall-at-5 0.85 --min-mrr 0.75
serve: ingest
	docqa serve
docker:
	docker build -t docqa-rag .
