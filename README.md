# Korean NLP RAG

[English](README.md) | [한국어](README.ko.md)

A local Korean evidence-retrieval and extractive question-answering system. It combines BM25 and dense retrieval, Cross-Encoder reranking and a KoELECTRA QA Reader to return an answer with a source citation or abstain.

The original four-week project was developed in June 2026. Its portfolio release adds task-aligned training, bounded model selection, a frozen project holdout and portable CPU execution, with reusable code and measured verification records.

## Engineering demonstrated

- Question-to-passage retriever training, filtered hard-negative mining and matched reranker refinements.
- BM25 and FAISS retrieval, Reciprocal Rank Fusion and Cross-Encoder reranking.
- Source-aligned natural chunks and retrieval-centered Reader context windows.
- Separate retrieval, answer quality, coverage and controlled no-answer measurements.
- Development-only selection, hash-bound final evaluation and retained rejected experiments.
- Verified `safetensors` artifacts, project-only environment setup and persistent CLI inference.

## Measured results

See [measured results](reports/results.md) for the paired final evaluation, development experiments, denominators, timing and limitations. [Historical regression](reports/legacy_baseline.md) is reported separately.

<!-- FINAL_RESULTS -->

| Measurement | Initial developed model | Final developed model | Change |
| --- | ---: | ---: | ---: |
| Answer EM, 501 answerable questions | 62.28% (312/501) | 73.25% (367/501) | +10.98 pp |
| Korean character F1, same 501 | 70.55% | 79.94% | +9.39 pp |
| Complete-span Recall@5, same 501 | 91.62% (459/501) | 96.61% (484/501) | +4.99 pp |
| Answerable coverage, same 501 | 86.03% (431/501) | 87.82% (440/501) | +1.80 pp |
| Abstentions, separate 80 synthetic questions | 79/80 (98.75%) | 76/80 (95.00%) | -3.75 pp |
| False answers, same 80 synthetic questions | 1/80 | 4/80 | +3 cases |
| Mean GPU serving time, all 581 | 105.34 ms | 209.53 ms | 1.99x |

Answer quality improves, but final synthetic false answers rise from 1 to 4 and mean GPU inference takes 1.99x as long. See the full report for the separate scopes and rejected experiments.

<!-- /FINAL_RESULTS -->

The final question set is a project-tuning holdout from public KorQuAD validation, not the official private test. Pretrained public models may have encountered that data. Controlled no-answer questions use fictional referents and do not measure natural-question hallucination rates.

## Run locally

Clone the [GitHub repository](https://github.com/Dongmyung378/Korean-NLP-RAG), or extract the [portfolio source archive](https://park-dongmyung.com/downloads/korean-nlp-rag-source.zip) if the repository is unavailable to your connection.

**Windows:** double-click `launch-rag.bat`, or run:

```powershell
.\launch-rag.bat
```

**Linux:**

```bash
bash launch-rag.sh
```

The launcher obtains 64-bit Python 3.12 when needed, creates a project-only environment, installs pinned CPU dependencies and downloads hash-verified artifacts. It reconstructs documents locally from an unchanged official KorQuAD download. Internet access is needed for the first setup; prepared model inference is local. Docker, an API key and a GPU are not required.

Plan for several gigabytes of free disk space for dependencies and models. Windows and Linux CPU execution are checked in isolated environments on the same host; this does not represent verification by an independent external person.

```powershell
.\launch-rag.bat --doctor
.\launch-rag.bat --question "방탄소년단의 팬클럽 창단식은 몇 월 며칠인가?" --json
```

The persistent question loop loads models once. `--question`, `--questions-file`, `--json` and `--no-reader` support single-query inference, batch evaluation and evidence inspection. See [runtime instructions](docs/runtime.md) for optional CUDA, offline verification and troubleshooting.

## Verify and reproduce

```bash
python -m unittest discover -s tests -v
python scripts/reproduce_data.py
```

Use the prepared project environment as explained in [runtime instructions](docs/runtime.md). The data helper downloads hash-pinned official train/dev sources and recreates the declared benchmark without redistributing transformed source material. Historical model-dependent tests skip when their optional original artifacts are absent.

The [evaluation guide](docs/evaluation.md) provides final inference commands and explains EM, Korean character F1, complete-span Recall and abstention. The [training guide](docs/training.md) records training commands and stopping rules. [Architecture](docs/architecture.md) explains how evidence and answer decisions connect.

## Repository map

| Location | Purpose |
| --- | --- |
| `korean_rag/` | Retrieval, extraction, metrics, training and CLI |
| `scripts/` | Setup, artifact verification, data reconstruction, training and evaluation |
| `configs/` | Immutable protocols, split IDs and selected configurations |
| `reports/` | Measured results, decision records and verification evidence |
| `docs/` | English guides and Korean translations |
| `tests/` | Input, artifact, split, selection and training checks |
| `artifacts/`, `.local/`, `.venv/` | Ignored local models, datasets and environment |

## Scope

The knowledge base contains 960 unique paragraphs from KorQuAD validation. The system targets factual questions within this fixed corpus. A citation proves that an extracted string occurs in its source; it does not prove semantic correctness. Thresholds are not calibrated correctness probabilities. Latest web information, arbitrary document ingestion and conversation memory are outside the implemented scope.

Upstream attribution is documented in [third-party notices](THIRD_PARTY_NOTICES.md). Dataset and model terms remain separate from this project's source code.
