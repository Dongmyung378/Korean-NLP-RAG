# Week 2 results

[English](week2_experiment_results.md) | [한국어](week2_experiment_results.ko.md)

## Objective

Train a Korean sentence-embedding model and extend it into a FAISS-based dense retriever. The question-context candidates and model files required by Week 3 were saved as well.

## Experiment results

| Configuration | Pearson | Spearman | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pretrained embedding baseline | 0.3475 | 0.3558 | 0.7318 | 0.9045 | 0.9545 | - |
| Final multi-seed model, seed 378 | 0.8519 | 0.8703 | 0.9500 | 0.9955 | 0.9955 | 0.9720 |
| KorQuAD transfer evaluation | - | - | 0.4930 | 0.7020 | 0.7880 | - |

## Summary

The pretrained correlation scores in the table come from the notebook's Day 1 output. Because a variable was reused, the `sts_baseline` key in the distributed `week2_metrics.json` contains the post-training Day 2 result (Pearson 0.7096, Spearman 0.7136). `sts_fine_tuned` contains the Day 3 result, while the final model correlations are recorded under seed 378 in `multi_seed_final_training`. This naming mismatch is not used by local model loading or inference.

In the controlled KLUE-STS retrieval evaluation, fine-tuning raised Recall@1 from 0.7318 to 0.9500. Seed 378 produced the highest correlation in the multi-seed experiment with hard negatives.

The comparison among seeds 17, 47, and 378 repeated only the 152-pair hard-negative refinement from the same Day 3 model. It did not repeat the complete training pipeline for each seed and therefore does not represent the seed variance of the full pipeline. The table reports the selected seed 378 result.

On the KorQuAD question-context evaluation, Top-20 Recall was 0.8560 and MRR@20 was 0.5878. Based on this result, Week 3 was designed to rerank the relevant context within the dense retriever's Top-20 candidates.

## Outputs

- Dense retriever checkpoint and tokenizer
- Retrieval configuration and FAISS index metadata
- 20,000 KorQuAD Top-20 candidates
- Reload verification results
