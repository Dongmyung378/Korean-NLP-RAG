# Notebook guide

[English](README.md) | [한국어](README.ko.md)

These notebooks record the four-week training, retrieval, and question-answering experiments. Each notebook runs in a fresh Kaggle session. Weeks 3 and 4 consume artifacts produced by earlier weeks, while the local application loads the final models and indexes directly.

## Weekly roles

| Week | Notebook | Main experiments | Role in the final application |
| --- | --- | --- | --- |
| 1 | [Transformer fine-tuning](<Week - 1/korean-transformer-finetuning.ipynb>) | KLUE-YNAT classification, configuration comparisons, hold-out evaluation | Training and evaluation exercise; classifier weights are not reused |
| 2 | [Sentence embedding](<Week - 2/sentence-embedding-retrieval.ipynb>) | KLUE-STS training, multi-seed refinement, FAISS retrieval | Dense retriever and tokenizer |
| 3 | [Cross-Encoder reranker](<Week - 3/cross-encoder-reranker.ipynb>) | Hard-negative training and ranking evaluation | Reranker checkpoint |
| 4 | [Hybrid RAG](<Week - 4/hybrid-rag.ipynb>) | BM25, dense retrieval, RRF, reranking, Qwen/reader comparison | Knowledge base, search indexes, and answer evaluation |

Weekly reports: [Week 1](../../reports/week1_experiment_results.md), [Week 2](../../reports/week2_experiment_results.md), [Week 3](../../reports/week3_experiment_results.md), [Week 4](../../reports/week4_experiment_results.md). Each report links to its Korean translation.

These learning notebooks are kept in version control for reproducibility, but their `.ipynb` files are excluded from GitHub's repository language statistics through `.gitattributes`.

## Kaggle execution order

1. Run each notebook from top to bottom in a fresh session with internet access and a training GPU.
2. Attach the Week 2 outputs as inputs to the Week 3 notebook.
3. Attach the Week 2 retriever and Week 3 reranker outputs to the Week 4 notebook.
4. After Week 4 finishes, download and extract `week4_final_artifacts.zip`.

In the Week 4 input cell, `WEEK2_CHECKPOINT` and `WEEK3_MODEL_DIR` default to `None`. The notebook selects an artifact automatically when exactly one candidate exists. If multiple versions are attached, execution stops and asks for an explicit full Kaggle path. The Week 2 configuration and tokenizer are loaded only from the selected checkpoint directory.

You do not need to rerun the notebooks to use the local CLI. Download the [prebuilt outputs from Google Drive](https://drive.google.com/drive/folders/1wqPSshOLzAJHYGfR7Z-Ug0BP_k3ycL4s?usp=sharing), place them under `transformer/outputs`, and follow the [main README](../../README.md).

## Week 4 result record

The Kaggle run produced gold-answer containment for 74 of 100 fixed questions. The reader candidate table contains 209 non-empty candidates. Every final answer in this sample came from the extractive reader, so the Qwen fallback was not used.

Single-character reader answers originally missed citations. Citation metadata, tables, and plots were recalculated locally from saved answers without changing any answer or rerunning model inference. Training and inference values remain the Kaggle records; citation metrics and related figures are local post-processing results. The exact scope is recorded in [week4_postprocessing.json](../../reports/week4_postprocessing.json).

The local CLI uses per-candidate thresholds and an explicit-year check without Qwen fallback. Its 67/100 containment result therefore represents a different answer policy from the notebook's 74/100 result.

## Interpretation notes

- The knowledge base consists of Korean Wikipedia contexts from the KorQuAD 1.0 validation split. It does not search the web.
- Answer containment checks normalized string inclusion, not semantic correctness. Citation rate is also not an accuracy metric.
- The Week 4 questions were used during development and do not constitute a new held-out benchmark.
- The `sts_baseline` field in the distributed Week 2 JSON contains the Day 2 trained result, despite its name. The true pretrained baseline and final seed result are separated in the [Week 2 report](../../reports/week2_experiment_results.md).
- Week 2 multi-seed runs repeat only the 152-pair refinement from the same Day 3 checkpoint, not the complete training pipeline.
- The Week 3 random-negative baseline saw some questions later used in its validation comparison. No controlled retraining with an identical question split was performed; see the [Week 3 report](../../reports/week3_experiment_results.md).

## Optional report regeneration

This step is not required for local question answering. Run it only to rebuild citations, tables, and figures from the saved Week 4 answers and candidates.

```bash
python -m pip install -r requirements-reports.txt
python scripts/refresh_week4_reports.py
```

The command updates Week 4 result files, post-processed notebook outputs, and the post-processing record without running model inference. Preserve a separate copy first if the original record must remain unchanged. The local `requirements.txt` covers the CLI, not every dependency used to train all four notebooks.
