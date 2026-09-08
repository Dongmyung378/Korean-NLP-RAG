# Korean NLP RAG

[English](README.md) | [한국어](README.ko.md)

[![Tests](https://github.com/Dongmyung378/Korean-NLP-RAG/actions/workflows/tests.yml/badge.svg)](https://github.com/Dongmyung378/Korean-NLP-RAG/actions/workflows/tests.yml)

A four-week NLP engineering project that builds a grounded Korean question-answering pipeline from trained retrieval and reranking models. The final application runs in a terminal, retrieves evidence from a fixed KorQuAD knowledge base, extracts an answer, and returns its citation or abstains when the evidence is insufficient.

This repository includes the experiment notebooks, reusable inference code, regression tests, evaluation records, and weekly reports. Large model artifacts are distributed separately through Google Drive.

## Highlights

- Trained a dense retriever on KLUE-STS with hard-negative refinement.
- Trained a Cross-Encoder reranker on question-context pairs.
- Combined BM25 and dense retrieval with Reciprocal Rank Fusion (RRF).
- Compared Qwen-based generation with an extractive QA reader.
- Added evidence citations, per-stage acceptance thresholds, and abstention behavior.
- Converted notebook experiments into a reusable Python CLI.
- Preserved evaluation inputs, predictions, environment versions, and source hashes.

## System architecture

```text
Korean question
      |
      +-- BM25 top 50
      +-- Dense retriever top 50
                 |
              RRF top 20
                 |
       Cross-Encoder reranker
                 |
               top 3
                 |
        Extractive QA reader
                 |
     answer + citation / abstain
```

The local application does not browse the web and does not use free-form generation. Qwen is retained only as a comparison experiment in the Week 4 notebook.

## Four-week project path

| Week | Work | Output used downstream |
| --- | --- | --- |
| 1 | KLUE-YNAT classification and training-configuration comparisons | Training and evaluation baseline; the classifier is not used by the RAG pipeline |
| 2 | KLUE-STS sentence embeddings, hard negatives, and FAISS retrieval | Dense retriever and tokenizer |
| 3 | Cross-Encoder training and ranking evaluation | Reranker checkpoint |
| 4 | BM25-dense fusion, reranking, reader/generator comparison, citations, and error analysis | Knowledge base, indexes, and final evaluation |

The notebooks and their execution order are documented in [transformer/notebooks](transformer/notebooks/README.md). Concise weekly reports are available under [reports](reports).

## Knowledge-base scope

The knowledge base contains 5,889 chunks created from 960 Korean Wikipedia contexts in the KorQuAD 1.0 validation split. It can answer some factual questions about the people, culture, history, and science represented in those contexts.

It cannot reliably answer questions about current news, weather, private information, or subjects absent from the saved corpus. A citation proves only that the extracted text occurs in the displayed evidence; it does not guarantee that the answer is semantically correct.

## Results

Retrieval was evaluated on 1,000 fixed questions. Answer generation and local inference were evaluated on a fixed 100-question development set.

| Metric | Result |
| --- | ---: |
| Hybrid + reranker Recall@5 | 0.902 |
| Hybrid + reranker Recall@20 | 0.935 |
| Notebook Reader-First answer containment | 74/100 |
| Local CLI answer containment | 67/100 |
| Local CLI accepted / abstained | 79 / 21 |
| Containment among accepted answers | 67/79 (84.8%) |
| Manual unanswerable probes abstained | 5/6 |

"Answer containment" means that a normalized gold-answer string occurs in the accepted answer. It is not semantic accuracy. The 100 questions were reused during development and are therefore a regression set, not a new held-out benchmark. The six unanswerable probes are a small debugging set that includes known failure cases.

See the [Week 4 report](reports/week4_experiment_results.md), [evaluation metrics](reports/local_evaluation/metrics.json), [100 predictions](reports/local_evaluation/predictions.json), and [unanswerable probes](reports/local_evaluation/unanswerable_probes.json) for the full record.

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/Dongmyung378/Korean-NLP-RAG.git
cd Korean-NLP-RAG
```

### 2. Download the model artifacts

Download the [`outputs` folder from Google Drive](https://drive.google.com/drive/folders/1wqPSshOLzAJHYGfR7Z-Ug0BP_k3ycL4s?usp=sharing), extract it if necessary, and place it under `transformer`.

```text
Korean-NLP-RAG/
  korean_rag/
  transformer/
    notebooks/
    outputs/
      week2/
        week2_retriever_artifacts/
          week2_dense_retriever.pt
          tokenizer/
      week3/
        week3_day4_final_reranker/
          config.json
          model.safetensors
      week4/
        week4_day1_3_artifacts/
          week4_knowledge_base_chunks.csv
          week4_dense.index
          week4_bm25.pkl
          week4_day5_rag_examples.csv
```

Avoid an extra `outputs/outputs` directory. Week 1 artifacts are not required by the local application. The checkpoints and BM25 index are deserialized at runtime, so use artifacts only from a trusted source.

### 3. Install dependencies and ask a question

```bash
python -m pip install -r requirements.txt
python -m korean_rag --question "방탄소년단의 팬클럽 창단식은 몇 월 며칠인가?"
```

Verified output:

```text
Answer: 2014년 3월 29일
Source: extractive_reader
Reader span score: 0.9591
Citation: 방탄소년단 (chunk 3167)
```

The first reader-enabled run downloads `monologg/koelectra-base-v3-finetuned-korquad` from Hugging Face. Later runs use the local cache. Each CLI invocation reloads the models.

The project was verified on Windows with Python 3.12.7, PyTorch 2.5.1+cu121, and Transformers 5.16.1. CPU inference is supported; CUDA requires a CUDA-enabled PyTorch installation.

## CLI examples

| Task | Command |
| --- | --- |
| Enter a question interactively | `python -m korean_rag` |
| Retrieve evidence without the QA reader | `python -m korean_rag --no-reader --question "질문"` |
| Run on CPU | `python -m korean_rag --device cpu --question "질문"` |
| Run on CUDA | `python -m korean_rag --device cuda --question "질문"` |
| Return JSON | `python -m korean_rag --json --question "질문"` |
| Use a different artifact directory | `python -m korean_rag --outputs-root "D:/data/outputs" --question "질문"` |
| Show all options | `python -m korean_rag --help` |

`--no-reader` shows retrieval and reranking results without extracting an answer. It is a diagnostic option, not a higher-accuracy mode.

Questions are limited to 1,000 characters and 128 model tokens. Inputs beyond either limit are rejected instead of silently truncated.

### Abstention behavior

```bash
python -m korean_rag --question "내일 서울 날씨는 어때?"
```

This out-of-scope question was abstained in the verified environment. The default acceptance thresholds are RRF 0.025, reranker 0.50, and reader 0.10. The pipeline also rejects empty spans, spans missing from the cited evidence, and answers whose evidence omits an explicit four-digit year in the question.

These scores are not calibrated correctness probabilities. Raising a threshold may reject correct answers, while a confidently wrong answer may still pass.

## Reproducing the checks

Run the lightweight tests without training or model inference:

```bash
python -m unittest discover -s tests -v
```

When the large artifacts are absent, artifact-dependent tests are skipped. GitHub Actions runs this lightweight suite on every push and pull request.

To repeat the 100-question local evaluation with the downloaded artifacts:

```bash
python -m korean_rag.evaluate --device auto --save-dir transformer/outputs/local_verification
```

This command performs inference but does not retrain any model. Use a different save directory if an existing evaluation should be preserved.

## Known limitations

- The fixed 100-question evaluation is a development/regression set, not an independent held-out benchmark.
- The six unanswerable probes are too small to measure general abstention performance. One false-premise question about a fan-club "dissolution ceremony" is still answered incorrectly.
- The Week 3 random-negative baseline was trained on some questions later used in its validation comparison. It cannot establish the causal benefit of hard-negative training. The final reranker ranking evaluation is reported separately in the [Week 3 report](reports/week3_experiment_results.md).
- The Week 2 multi-seed experiment repeats only the 152-pair hard-negative refinement from a shared checkpoint, not the complete training pipeline.
- The answer-in-evidence check is string-based and cannot verify semantic entailment.
- Stored evaluation hashes identify the code used for that recorded run. Input validation and exception handling were added afterward without rewriting the historical result files.

## Repository layout

```text
korean_rag/            Local retrieval, reranking, QA, CLI, and evaluation code
transformer/notebooks/ Four weekly Kaggle experiment notebooks
reports/               Weekly analyses and saved local evaluation records
tests/                 Unit, notebook, and artifact regression tests
scripts/               Report post-processing utility
```
