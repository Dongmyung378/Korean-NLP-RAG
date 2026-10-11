# Development abstention calibration

[한국어](development-calibration.ko.md) | [Machine-readable audit](development-calibration.json)

The selected development configuration answers 228 of 304 questions exactly (75.00% EM), achieves 83.01% Korean character F1 and answers 278/304 answerable questions (91.45% coverage). It falsely answers 1/80 synthetic no-answer questions, matching the unchanged original model measured on the same GPU.

| Development measure | Original model | Selected development configuration | Difference |
| --- | ---: | ---: | ---: |
| Exact match | 188/304, 61.84% | 228/304, 75.00% | +13.16 percentage points |
| Character F1 | 68.92% | 83.01% | +14.09 percentage points |
| Answerable coverage | 257/304, 84.54% | 278/304, 91.45% | +6.91 percentage points |
| Synthetic false answers | 1/80, 1.25% | 1/80, 1.25% | Unchanged |

The original four-Reader comparison selected five parent windows with Reader confidence multiplied by reranker confidence for answer ranking. However, all twelve original threshold candidates failed the joint quality, coverage and safety rule. Those sixteen records remain unchanged and checksum-bound. A separate declaration, written before follow-up scoring, authorized exactly nine stricter threshold candidates using the same verified 384-question development cache. This explicitly expands the budget from twelve to twenty-one threshold configurations. The original benchmark and validation protocol were unchanged; no final questions were used.

The follow-up crossed RRF thresholds 0.028, 0.030 and 0.032 with Reader confidence thresholds 0.3, 0.6 and 0.9. Reranker threshold 0.5, explicit-year checking, the selected Reader and its fixed `1e-6` near-null guard remained unchanged. The original same-GPU false-answer rate and coverage constraints remained mandatory. Six candidates passed; three failed coverage. Quality was ranked by EM, then F1, then lower false-answer rate and declared order.

| Candidate | RRF | Reader | EM | Character F1 | Answerable coverage | False answers | Joint promotion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0, selected | 0.028 | 0.3 | 228/304, 75.00% | 83.01% | 278/304 | 1/80 | Pass |
| 1 | 0.028 | 0.6 | 226/304, 74.34% | 82.00% | 272/304 | 1/80 | Pass |
| 2 | 0.028 | 0.9 | 223/304, 73.36% | 80.19% | 265/304 | 0/80 | Pass |
| 3 | 0.030 | 0.3 | 224/304, 73.68% | 81.11% | 269/304 | 0/80 | Pass |
| 4 | 0.030 | 0.6 | 222/304, 73.03% | 80.10% | 263/304 | 0/80 | Pass |
| 5 | 0.030 | 0.9 | 219/304, 72.04% | 78.16% | 256/304 | 0/80 | Pass |
| 6 | 0.032 | 0.3 | 211/304, 69.41% | 74.72% | 241/304 | 0/80 | Fail coverage |
| 7 | 0.032 | 0.6 | 209/304, 68.75% | 73.71% | 236/304 | 0/80 | Fail coverage |
| 8 | 0.032 | 0.9 | 205/304, 67.43% | 71.63% | 227/304 | 0/80 | Fail coverage |

This is an adaptive development result. Reusing the development set can produce optimistic selected estimates, and the synthetic no-answer cases cover only their controlled fictional construction. The [completed final paired evaluation](final/final-comparison.json) was performed after artifact and configuration freeze and is reported separately: final answer quality improved, while synthetic false answers increased from 1/80 to 4/80. Neither development promotion nor the development one-false-answer count establishes general safety, semantic correctness or private-benchmark performance.
