# Evaluation report

46 golden questions over 53 chunks.

## Retrieval ablation (answerable questions)

| Mode | Recall@1 | Recall@3 | Recall@5 | MRR@10 | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| bm25 | 0.779 | 0.912 | 0.941 | 0.858 | 0.889 | 0.11 | 0.21 |
| dense | 0.779 | 0.912 | 0.941 | 0.858 | 0.889 | 0.27 | 0.36 |
| hybrid | 0.809 | 0.941 | 0.941 | 0.875 | 0.902 | 0.46 | 1.28 |
| hybrid_rerank | 0.779 | 0.912 | 0.941 | 0.858 | 0.889 | 1.51 | 2.84 |

## Recall@5 by category

| category | n | bm25 | dense | hybrid | hybrid_rerank |
|---|---|---|---|---|---|
| lexical | 10 | 1.000 | 1.000 | 1.000 | 1.000 |
| multi | 2 | 0.500 | 0.500 | 0.500 | 0.500 |
| paraphrase | 22 | 0.955 | 0.955 | 0.955 | 0.955 |

## Recall@5 by split

| split | n | bm25 | dense | hybrid | hybrid_rerank |
|---|---|---|---|---|---|
| dev | 12 | 1.000 | 1.000 | 1.000 | 1.000 |
| test | 22 | 0.909 | 0.909 | 0.909 | 0.909 |

## Answers (generator: `extractive`, retrieval: `hybrid`, min_support=0.5)

| Metric | Value |
|---|---|
| Keyword coverage (answerable, n=34) | 0.618 |
| Citation precision | 0.729 |
| False abstention rate (answerable) | 0.294 |
| Correct abstention rate (unanswerable, n=12) | 0.667 |

## Support-threshold sweep

**dev** (answerable n=12, unanswerable n=6)

| min_support | False abstention (lower is better) | Correct abstention (higher is better) |
|---|---|---|
| 0.2 | 0.000 | 0.167 |
| 0.25 | 0.000 | 0.167 |
| 0.3 | 0.000 | 0.333 |
| 0.35 | 0.000 | 0.500 |
| 0.4 | 0.000 | 0.500 |
| 0.5 | 0.083 | 0.667 |
| 0.6 | 0.333 | 0.833 |

**test** (answerable n=22, unanswerable n=6)

| min_support | False abstention (lower is better) | Correct abstention (higher is better) |
|---|---|---|
| 0.2 | 0.000 | 0.000 |
| 0.25 | 0.000 | 0.167 |
| 0.3 | 0.045 | 0.333 |
| 0.35 | 0.227 | 0.500 |
| 0.4 | 0.227 | 0.500 |
| 0.5 | 0.409 | 0.667 |
| 0.6 | 0.545 | 1.000 |

**all** (answerable n=34, unanswerable n=12)

| min_support | False abstention (lower is better) | Correct abstention (higher is better) |
|---|---|---|
| 0.2 | 0.000 | 0.083 |
| 0.25 | 0.000 | 0.167 |
| 0.3 | 0.029 | 0.333 |
| 0.35 | 0.147 | 0.500 |
| 0.4 | 0.147 | 0.500 |
| 0.5 | 0.294 | 0.667 |
| 0.6 | 0.471 | 0.917 |

Threshold chosen on dev only (largest value with <=10% false abstention): **0.5**.

## Retrieval failures for `hybrid` (required reference not in top 5)

| id | category | question | recall@5 |
|---|---|---|---|
| q29 | paraphrase | Under what conditions is a canary reverted without human action? | 0.00 |
| q34 | multi | What actions are needed after failing over to the secondary region? | 0.00 |
