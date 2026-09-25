# Week 3 results

[English](week3_experiment_results.md) | [한국어](week3_experiment_results.ko.md)

## Objective

Use the Week 2 dense-retrieval output to train a Cross-Encoder reranker that reads each question and context together. Classification scores and ranking metrics were recorded separately.

## Experiment results

| Configuration | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Week 2 dense retrieval | 0.4610 | 0.6460 | 0.7380 | 0.8180 | 0.5437 | 0.5899 |
| Dense retrieval + reranker | 0.7520 | 0.8100 | 0.8160 | 0.8180 | 0.7796 | 0.7887 |

In the learning-rate comparison, 3e-5 achieved the highest validation Macro-F1 of 0.9732. The saved final checkpoint was subsequently retrained with a learning rate of 1e-5 and weight decay of 0.01, reaching validation Macro-F1 0.9750. Week 4 used this saved checkpoint.

## Summary

### Limitation of the baseline comparison

The Day 3 random-negative baseline was trained on questions that included the hard-negative target questions. On Day 4, question IDs were split into training and validation sets, and that same baseline was evaluated on the validation set, meaning it had already seen some validation questions. The difference between `hard_negative_baseline_macro_f1` and the Day 4 model therefore should not be interpreted as independent validation or conclusive evidence of the benefit of hard-negative training.

The final model reloaded the pretrained weights and trained on the Day 4 training split. The ranking table above was measured on separate KorQuAD validation queries and should be distinguished from the question-overlap issue in the baseline comparison. No controlled experiment trained random-negative and hard-negative models on the same question split.

The reranker does not expand the Top-20 candidate set, but it substantially improves the ordering within that set. The unchanged Recall@20 shows that Week 2 retrieval is responsible for candidate generation, while the Week 3 model improves the quality of the top ranks.

The largest gains are in Recall@1 and MRR@10. These results motivated the Week 4 design that reranks the Top-20 candidates from hybrid retrieval.

## Outputs

- Hard-negative training data
- Final Cross-Encoder checkpoint
- Candidate file containing both dense and reranker ranks
- Ranking-evaluation results and error examples
