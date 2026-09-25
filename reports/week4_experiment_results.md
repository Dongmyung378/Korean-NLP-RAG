# Week 4 results

[English](week4_experiment_results.md) | [한국어](week4_experiment_results.ko.md)

## Evaluation scope

This report covers the retrieval and answering experiments run on Kaggle and the local application evaluation. Citation metrics were calculated through local post-processing of saved answers and are recorded separately from inference results.

Retrieval was evaluated on 1,000 fixed questions over 5,889 chunks derived from 960 KorQuAD contexts. Answer methods were compared on a fixed subset of 100 questions. Because this sample was already used during development, it is not a new held-out evaluation.

## Retrieval results

| Method | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.6090 | 0.7860 | 0.8240 | 0.8510 | 0.6870 | 0.6958 |
| Dense retrieval | 0.5490 | 0.7950 | 0.8510 | 0.9010 | 0.6535 | 0.6767 |
| Hybrid RRF | 0.6620 | 0.8590 | 0.9000 | 0.9350 | 0.7452 | 0.7604 |
| Hybrid RRF + reranker | 0.6880 | 0.9020 | 0.9220 | 0.9350 | 0.7795 | 0.7942 |

BM25 was strongest at rank one, while dense retrieval was stronger across the wider candidate range. Combining both retrievers with RRF and applying the reranker improved the top-ranked results. Recall@20 remained unchanged because the reranker only reordered the same candidate set.

## Notebook answer results

| Method | Answer containment | Evidence-string match | Answer-stage time/query |
| --- | ---: | ---: | ---: |
| Qwen Direct Top-3 | 0.53 | 0.78 | 0.459 s |
| Qwen Question-Focused Top-3 | 0.53 | 0.78 | 0.448 s |
| Qwen Focused + Validation | 0.55 | 0.89 | 0.558 s |
| Extractive QA Reader | 0.74 | 1.00 | 0.031 s |
| Reader-First final answer | 0.74 | 1.00 | 0.031 s |

Answer containment checks whether the normalized gold-answer string occurs in the response. There were 209 non-empty reader candidates, and reader answers were selected for all 100 questions. The Qwen fallback was not used in this sample.

The preceding run produced an answer-containment score of 0.64. Because the reranking results and runtime environment also changed, the increase to 0.74 cannot be attributed only to the change in empty-answer handling.

Times are averages for the answer stage measured on a Kaggle Tesla T4. They exclude model loading, retrieval, and reranking. Qwen generation has a separate cost as well, so 0.031 seconds must not be presented as the end-to-end application response time.

## Citation metric calculation

A reader citation points to the chunk from which the answer was actually extracted. In the original run, a condition that excluded single-character answers caused citations to be omitted for answers such as `진` and `열`. Post-processing allowed single-character reader answers without changing Qwen's validation condition.

Citations, tables, and figures were recalculated locally using only the saved answers and candidates. No answers were changed and no GPU inference was run. Evidence-string match and citation rate changed from 0.97 to 1.00. Incorrect extractions such as `고,` also occur in their source text, so a 100% citation rate is not 100% answer accuracy.

The post-processed notebook cells contain `local_postprocessing` metadata. The [post-processing record](week4_postprocessing.json) stores input and output hashes and the exact scope of changes; the original Kaggle inference times were preserved.

## Remaining errors

| Category | Questions |
| --- | ---: |
| Gold-answer string contained | 74 |
| Evidence retrieved, but reader extraction failed | 15 |
| Gold answer absent from Top-3 evidence | 11 |

Because the experiment used answerable questions, these results alone cannot evaluate abstention on unanswerable questions.

## Local application evaluation

| Metric | Result |
| --- | ---: |
| Gold-answer string contained | 67/100 |
| Answers returned | 79/100 |
| Abstentions | 21/100 |
| Containment among returned answers | 67/79 (84.8%) |
| Mean query time, excluding loading | 0.084 s |
| Model loading time | 13.99 s |
| Manual unanswerable probes abstained | 5/6 |

The local application applies per-candidate thresholds of RRF 0.025, reranker 0.50, and reader 0.10, plus an explicit-year check, without a Qwen fallback. The notebook's 74% and the local application's 67% therefore reflect different answer policies.

Retrieval through answering was rerun using Python 3.12.7, PyTorch 2.5.1+cu121, Transformers 5.16.1, and CUDA. The repository stores the [metrics and input hashes](local_evaluation/metrics.json), [100 predictions](local_evaluation/predictions.json), and [unanswerable probe results](local_evaluation/unanswerable_probes.json).

Of the six unanswerable probes, the application abstained on weather, private information, a future performance, an incorrect year, and a meal question absent from the corpus. It returned a date for the false premise about a fan-club "dissolution ceremony." This is a small debugging set that includes known failure cases and should not be interpreted as a general abstention benchmark.

## Reproduction commands

- Verify code and saved results: `python -m unittest discover -s tests -v`
- Rerun local inference evaluation: `python -m korean_rag.evaluate --device auto --save-dir reports/local_evaluation_rerun`
- Recalculate citations and tables without model inference: `python scripts/refresh_week4_reports.py`

The report-regeneration command updates result files and post-processed notebook outputs. See the [notebook README](../transformer/notebooks/README.md) for runtime requirements and guidance on preserving the original record.
