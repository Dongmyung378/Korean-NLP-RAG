# Measured portfolio release results

[English](results.md) | [한국어](results.ko.md)

The final development-selected system answers 367 of 501 held-out answerable questions exactly, compared with 312 for the initial system. Retrieval and answer quality improve, while synthetic no-answer errors increase and inference takes longer. These tradeoffs are part of the result, and final outcomes were not used to retune or reselect the configuration.

## Reading the comparison

“Paired” means both implementations answer the same questions, allowing changes to be inspected question by question. “Frozen” means the chosen configuration and its file identities were recorded before final inference. Development questions informed selection; the final holdout did not, subject to the historical exposure limits below.

EM checks exact answer matching after normalization, while character F1 measures partial overlap with the reference answer. Recall@5 evaluates the search stage: at least one of the top five candidates must contain a complete gold answer span, where “gold” means the supplied reference annotation. Coverage counts returned answers whether correct or wrong. Synthetic abstention measures withheld answers on a separate controlled set with no supported answer. It cannot be added to answerable-question accuracy as though all rows shared one denominator.

For timings, p50 is the median observed request time and p95 is the time at or below which approximately 95% of measured requests fall. Cold loading is the one-time model startup cost, distinct from serving questions after loading.

## Frozen paired final evaluation

| Measurement | Initial developed model | Final developed model | Change |
| --- | ---: | ---: | ---: |
| Answer EM, 501 answerable questions | 62.28% (312/501) | 73.25% (367/501) | +10.98 pp |
| Korean character F1, same 501 | 70.55% | 79.94% | +9.39 pp |
| Complete-span Recall@5, same 501 | 91.62% (459/501) | 96.61% (484/501) | +4.99 pp |
| Answerable coverage, same 501 | 86.03% (431/501) | 87.82% (440/501) | +1.80 pp |
| Abstentions, separate 80 synthetic questions | 79/80 (98.75%) | 76/80 (95.00%) | -3.75 pp |
| False answers, same 80 synthetic questions | 1/80 | 4/80 | +3 cases |
| Mean GPU serving time, all 581 | 105.34 ms | 209.53 ms | 1.99x |

Both systems ran the same frozen 501 answerable questions and separate 80 synthetic no-answer questions on Windows 11 with Python 3.12.7, PyTorch 2.14.0+cu130 and an NVIDIA RTX 3060 Laptop GPU. GPU serving time excludes model loading and includes retrieval, reranking and the configured Reader path. Mean, median and p95 were 105.34/103.38/125.02 ms initially and 209.53/206.98/247.23 ms finally; cold loading took 15.55 and 19.10 seconds. These are single-host observations during development, not dedicated throughput or universal latency guarantees.

The comparison is a project-tuning holdout constructed from public KorQuAD validation. It is not the official private test, and public pretrained models may have encountered those documents or questions. Nineteen articles occur in both development and final splits, although paragraph and normalized-question groups are separated. F1 uses official-compatible Korean character normalization. Recall requires a retrieved chunk in the correct paragraph containing an entire gold span; its counts refer to reranked candidates.

## Improvements and stopped experiments

1. **An auditable evaluation boundary.** Separate whole-paragraph question groups define 304 development and 501 final answerable cases. Historical 1,000-query exposure and known original-training groups are excluded from final selection. All 960 unique corpus paragraphs remain searchable. Dataset reconstruction reproduces five recorded file hashes, and the final freezes bind data, model configuration, bundle manifest and nineteen source/dependency files before inference.
2. **Task-aligned training with filtered negatives.** A 3,002-question training subset and 206-question internal validation subset were built from official training data. Question-to-passage retriever training raised internal Recall@5 from 96.12% to 100% at epoch one; epoch two fell to 99.03%, so epoch one was retained. Matched random-negative and hard-negative reranker refinements each ran one epoch from the same already-trained control. Their internal metric remained at its original 100% ceiling, so both refinements stopped and retained the exact original weights. The ceiling and uncertain historical control exposure limit this experiment's conclusions. It provides no demonstrated superiority of one negative strategy.
3. **End-to-end retrieval selection.** Six development configurations compared the initial system, task-trained retrieval and public E5 with several source-only chunk sizes. The final retrieval configuration uses multilingual E5 with the proper query/passage prefixes, masked mean pooling and normalized vectors, 192-token chunks and 32-token overlap. It creates 2,264 chunks versus 5,889 initially. E5 with longer chunks and query limits is a compound change; the final gain cannot be attributed solely to model training. The trained retriever remains an optional reproducible artifact, while the stronger joint development configuration is used for serving.
4. **Source-aligned Reader windows and explicit acceptance gates.** Four Reader paths were compared on development questions. The selected path considers five candidates, reconstructs retrieval-centered parent windows within a 512-token budget and combines Reader and reranker scores. It uses no gold answer offsets at inference. A fixed near-null margin and source citation checks accompany thresholds. All twelve initial threshold combinations failed the joint development gate. A separately declared nine-candidate stricter follow-up retained every failed record and unchanged quality, coverage and same-GPU safety criteria; six passed. The selected development result was EM 75.00%, F1 83.01%, coverage 91.45% and 1/80 synthetic false answers. The effective threshold budget is 12 plus 9, an explicitly adaptive development refinement.
5. **Portable execution and reproducible artifacts.** The public launcher prepares pinned Python and CPU dependencies inside the project, validates inventory and hashes, reconstructs source text locally, loads only local safetensors for serving and keeps models loaded for repeated questions. Windows and Linux automatic Python bootstrap were actually executed. Frozen final CPU execution was checked in separated environments on the same host: two predeclared development questions, each repeated twice, yielded eight matching answers, decisions and citation identities. Model loading was 14.44 seconds on Windows and 11.83 on Linux, with observed peak process memory 2.05 and 1.64 GiB. This is an environment smoke test, not a full CPU benchmark or external independent review. See [environment verification](environment-verification.md).
6. **A reviewable portfolio.** Reusable training, selection, evaluation and runtime code replace personal learning notebooks in the public tree. English guides have Korean companions. Initial-model conversion, metadata/source archives, compact final predictions, attribution and rejected development outcomes are documented. The original STS retriever and task-trained candidate are published in an optional [training-model archive](training-artifacts.md), verified by a complete anonymous download and SHA-256. The selected E5 runtime uses separate artifacts and does not depend on this archive.

## Final changes and remaining errors

The bootstrap below repeatedly samples the recorded question groups and recalculates the difference using both systems' saved answers. Its seed fixes the random resampling sequence; it does not refer to retraining the models. The resulting interval describes the observed sample under those grouping assumptions.

The paired final answers contain 82 newly exact-correct cases and 27 lost exact-correct cases, for a net gain of 55. F1 improves on 98 questions and falls on 38. A seed-378 bootstrap over 102 connected paragraph/question groups, with 2,000 resamples, gives descriptive paired 95% intervals of +6.94 to +15.26 percentage points for EM and +5.35 to +13.37 for F1. Residual article dependence, pretrained exposure and adaptive development selection are outside that calculation; it is not a general statistical-significance claim.

The final system still abstains on 61 answerable questions and accepts 73 non-exact answers. Among cases with a full gold span in the top five retrieved chunks, 51 are abstained and 68 are answered incorrectly. These positions identify review targets rather than prove causal component failures: parent windows can extend beyond a retrieved chunk. A valid citation establishes source occurrence, not that the answer matches the requested relation.

No-answer performance is a clear limitation. Four new false answers occur in the fictional-entity identity subtype (4/20), while the initial model's one fabricated-relation error is corrected. The total therefore increases from 1/80 to 4/80. The development safety condition does not generalize unchanged to final questions. The selected configuration remains the development-selected default; the more conservative initial system is separately reproducible. No threshold was changed after observing these outcomes. These deliberately constructed closed-corpus cases do not estimate error rates on natural user questions.

The frozen selected model was also run on all 80 synthetic final questions in the separate Windows CPU environment. It produced the same four false-answer IDs as GPU, with zero answer/abstain decision mismatches. This supplemental device audit used the unchanged configuration and does not add a full 501-question CPU quality measurement. See [synthetic CPU audit](synthetic-final-cpu-audit.md).

## Verify the published result

The full unit suite ran in isolated pure CPU environments on both Windows and Linux: **112 tests per operating system, 109 passed and 3 optional historical-artifact checks skipped**. Linux used read-only source and disabled networking. Two test scratch paths were made portable without changing assertions; the affected fifteen tests also passed again on Windows. The suite includes a real tiny optimizer run, corrupted-artifact rejection, data separation, source offsets, selection budgets and environment isolation. Both final runs passed independent evidence/hash audits and saved-prediction rescoring over all 581 question IDs. The final audit replays selection from saved Reader outputs and performs no new inference.

After publication, a fresh source archive and empty model cache passed the actual default Windows CPU setup using only credential-free public downloads. Preparation took 223.43 seconds; six demo inferences across two persistent sessions confirmed repeat decisions and offline reuse. Both demo questions were abstained, so this measures delivery and execution rather than answer accuracy. The exact initially tested archive and unchanged nineteen-file core are recorded in [public download verification](public-download-verification.md). GitHub rejected the hosted workflow dispatch because Actions is disabled for the account; [CI status](ci-status.md) distinguishes that restriction from the completed local checks.

See [machine-readable comparison](final/final-comparison.json), [initial metrics](final/final-original-metrics.json), [final metrics](final/final-selected-metrics.json), [development retrieval](development-retrieval.md), [development calibration](development-calibration.md), [test record](verification-tests.json) and [runtime instructions](../docs/runtime.md). Compact predictions omit source passages, questions and gold answers. Their checksums are distinct from the private full-evidence checksums.

After preparing the environment and selected bundle, reconstruct the benchmark from unchanged official downloads and run a new final evaluation directory:

```powershell
.venv\Scripts\python.exe scripts\reproduce_data.py --output-dir .local\benchmark-v1
.venv\Scripts\python.exe scripts\evaluate_portable.py --bundle artifacts\runtime --benchmark .local\benchmark-v1 --split final --no-answer reports\no-answer-v1\final.jsonl --freeze configs\selected-freeze.json --device cpu --output .local\reproduced-final
```

Linux uses `.venv/bin/python` and slash paths. Reproducing the measured GPU scores uses the explicitly prepared CUDA distribution and `--device cuda`; processor/backend differences can affect floating-point decisions. Use the [documented original conversion and bundle preparation](../docs/runtime.md#explicit-preparation-of-the-original-project-retriever), `artifacts/original`, `configs/original-freeze.json` and a different output directory for the original comparison. Data, source and model hashes identify what was measured; they do not promise bitwise equality on every machine.
