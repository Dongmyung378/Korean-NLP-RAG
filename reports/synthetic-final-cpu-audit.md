# Supplemental synthetic final CPU audit

[English](synthetic-final-cpu-audit.md) | [한국어](synthetic-final-cpu-audit.ko.md)

The frozen selected model was additionally run on all 80 synthetic final no-answer questions in a separate clean Windows CPU environment after the GPU final evaluation. No answerable holdout questions were inferred on CPU, and no policy tuning or model selection followed this audit.

| Measurement | GPU final | Supplemental CPU audit |
| --- | --- | --- |
| Synthetic no-answer questions | 80 | 80 |
| False answers | 4 | 4 |
| False-answer rate | 5.00% | 5.00% |

There were 0 CPU/GPU decision disagreements. The source, configuration, model manifest, freeze and question-file hashes were checked. These synthetic cases represent a limited no-answer stress test, not universal answer safety or external independent review. The complete answerable final benchmark remains the GPU measurement. CPU loading took 12.44 s; mean inference time across these 80 synthetic cases was 2.99 s on this host.

[synthetic-final-cpu-audit.json](synthetic-final-cpu-audit.json) publishes only query IDs, decisions, hashes and aggregate environment measurements. Full predictions remain private. The separate two-question Windows/Linux portability smoke is described in [environment-verification.md](environment-verification.md).
