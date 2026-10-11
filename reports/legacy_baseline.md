# Original development model

[한국어](legacy_baseline.ko.md)

The original four-week project was developed in June 2026. Its implementation is retained by source revision `de511597bf7556076aaa4c917ddcc12dd98c041e`. The original pipeline uses the project's STS-trained KLUE-RoBERTa retriever, trained two-class reranker, 5,889 short chunks, BM25, FAISS, RRF and the published KorQuAD-finetuned KoELECTRA reader.

Historical search results used 1,000 development questions: hybrid plus reranker Recall@5 90.2%, Recall@20 93.5% and MRR@10 0.7795. They are historical development results and have a different question set from the new evaluation.

The saved 100-question local regression run accepted 79 answers. Its previous gold-string containment count was 67/100, or 67/79 among accepted answers. Rescoring those unchanged saved predictions with official KorQuAD normalization gives EM 63/100, character F1 68.54% and accepted-answer EM 63/79 (79.75%). Containment, exact match and semantic correctness are different measurements.

The six historical no-answer probes produced five abstentions. The dissolution-ceremony question was incorrectly answered with a founding-ceremony date. These previously inspected debugging cases are reported separately from new synthetic no-answer evaluation.

The original random-negative training comparison had evaluation overlap. New random/hard refinement arms therefore use identical questions, positives, initialization and schedules, with paragraph-separated refinement validation. Both start from the same original hard-trained checkpoint; this controls the additional refinement experiment and does not establish the effect of pure training from the base model.

Detailed original notebooks and source exports are preserved privately. Public reports focus on reproducible configuration, measured results and limitations. Maintenance dates do not change the original June development period.
