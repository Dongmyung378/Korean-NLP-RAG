# Evaluation

[한국어](evaluation.ko.md)

Evaluation uses a frozen project holdout from public KorQuAD v1 validation. Whole source paragraphs and normalized duplicate questions stay in the same group. The final split excludes groups exposed in the historical 1,000-question evaluation and original training artifacts. All 960 source paragraphs remain searchable, including final gold documents.

| Defined set | Questions |
| --- | ---: |
| Available official-training records after overlap removal | 60,387 |
| Answerable development | 304 |
| Answerable final | 501 |
| Synthetic no-answer development | 80 |
| Synthetic no-answer final | 80 |

The complete comparisons contain 384 development and 581 final questions. Targets of 300 and 500 answerable questions were exceeded to preserve whole groups. The bounded training experiment uses 3,002 questions and a separate 206-question internal validation subset drawn entirely from the official training source. Project development and final questions are excluded from that training preparation.

Exact Match and character F1 follow the official Korean KorQuAD normalization and take the best score over gold annotations. Abstaining on an answerable question scores zero. Retrieval hit Recall, MRR and nDCG require a chunk from the correct paragraph containing an entire annotated answer span. Partial overlap does not qualify. Report answerable coverage, answered-answer accuracy and accepted-answer precision separately; unsupported synthetic answers count as false accepts. Wilson intervals are descriptive and do not establish significance for correlated or synthetic questions.

Reader confidence is the product of start/end token probabilities, used as an inference selection signal. It is not a calibrated probability that an answer is semantically correct. A cited span appearing in source text alone does not establish that it answers the requested subject or relation.

Selection uses development questions only: retrieval gain must reach one percentage point; answer EM gain must reach one point, or character F1 gain must reach half a point without an EM regression. Budgets allow four chunking configurations, four Reader configurations and twelve threshold combinations. Threshold promotion also requires no-answer false-answer rate no higher than the original control and answerable coverage at least original coverage minus two points. A fixed `1e-6` span-versus-CLS margin applies to every candidate Reader, while the unchanged control keeps margin zero. Compare control and candidate on the same device.

The initial twelve threshold candidates all failed joint promotion. A separate [calibration-v2 declaration](../configs/calibration-v2.json) then bounded a follow-up to nine stricter combinations on the same development cache, keeping the selected Reader and every original quality, coverage and same-GPU safety gate fixed. This expands the threshold budget to 12 initial plus 9 follow-up candidates and is explicitly an adaptive development refinement, not success under the original budget. Every failed initial record remains unchanged. Six follow-up candidates passed; the highest-EM candidate was selected. See the [complete development calibration report](../reports/development-calibration.md). These selected development estimates can be optimistic; final evaluation is performed only after configuration freeze and reported separately.

The completed [paired final audit](../reports/final/final-comparison.json) improved answerable EM from 312/501 to 367/501 and character F1 from 70.55% to 79.94%. Synthetic false answers increased from 1/80 to 4/80. The development safety constraint did not generalize to this synthetic final set, and these final outcomes were not used to retune thresholds. The original configuration remains reproducible for comparison. Descriptive paragraph-group bootstrap intervals accompany paired gains and losses; they do not remove public pretrained exposure or residual article dependence.

The public E5 encoder with longer chunks and query limits is a compound system candidate. Improvement in that comparison cannot be attributed solely to encoder training. Both systems use the identical corpus, question manifests, gold relevance rules and unchanged reranker/Reader controls before the bounded Reader comparison.

Before final access, freeze the selected configuration, both benchmark bindings, complete bundle manifest and every inference/metric source hash. The final command rejects variant caches and incomplete or changed freezes before loading models or reading final questions. Preserve one predefined paired original/selected comparison and every rejected development experiment. Final outcomes must never select a new configuration.

```powershell
python -m scripts.evaluate_portable --bundle PATH_TO_BUNDLE --benchmark PATH_TO_BENCHMARK --split dev --no-answer PATH_TO_NO_ANSWER/dev.jsonl --output NEW_DEV_OUTPUT --device cpu
python -m scripts.evaluate_portable --bundle PATH_TO_FROZEN_BUNDLE --benchmark PATH_TO_BENCHMARK --split final --no-answer PATH_TO_NO_ANSWER/final.jsonl --freeze PATH_TO_FREEZE_JSON --output NEW_FINAL_OUTPUT --device cpu
```

Each run saves prediction and Reader-cache checksums, question/benchmark bindings, measured environment settings and metrics with counts. Models load locally from a verified artifact bundle; inference does not require Docker. Reproduction of hashes does not promise bitwise equality across devices. Serving latency excludes model loading, and development caches evaluating multiple Reader paths have a separate latency scope.

This holdout is neither the official private test nor proof of pretrained-model independence. Nineteen source articles overlap development and final, and published KorQuAD-trained models may have encountered public validation data. The [no-answer set](../reports/no_answer_methodology.md) is an auditable synthetic closed-corpus stress test, with six previously exposed probes kept outside its final denominator. Confirmed numerical outcomes belong in the saved result reports; this document assigns no unmeasured final scores.
