# Week 1 results

[English](week1_experiment_results.md) | [한국어](week1_experiment_results.ko.md)

## Objective

Fine-tune KLUE-RoBERTa for KLUE-YNAT news-title classification and examine the balance between performance and training cost.

## Experiment results

| Configuration | Evaluation split | Accuracy | Macro-F1 | Parameters | Training time |
| --- | --- | ---: | ---: | ---: | ---: |
| Fixed-seed baseline | Dev | 0.8831 | 0.8693 | 110.6M | 235.0 s |
| Learning rate 3e-5 | Dev | 0.8827 | 0.8686 | 110.6M | 236.9 s |
| Selected final model | Hold-out test | 0.8696 | 0.8689 | 110.6M | 709.3 s |

## Summary

The fixed-seed baseline achieved the highest Dev Macro-F1 of 0.8693 and was used for final training. The final hold-out test Macro-F1 was 0.8689.

Because the news titles were short, sequence lengths above 64 provided little benefit. Errors were concentrated among closely related categories such as politics, society, economy, and culture/lifestyle.

## Outputs

- Final classification model
- Hold-out test predictions and error examples
- Training configurations and evaluation metrics
