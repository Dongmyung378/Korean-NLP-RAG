# Environment verification

[English](environment-verification.md) | [한국어](environment-verification.ko.md)

The frozen selected model passed offline CPU smoke tests in separate clean Windows and Linux environments. Both imported the same frozen source and used the same verified artifacts. Two predeclared development questions were each executed twice in each environment, for eight inferences in total. The answer text, answer/abstention decisions and citation identities matched across operating systems and repeats.

| Measurement | Windows | Linux |
| --- | --- | --- |
| Python | 3.12.7 | 3.12.15 |
| PyTorch | 2.14.0+cpu | 2.14.0+cpu |
| CPU threads | 4 | 4 |
| Inferences completed | 4 | 4 |
| Model loading | 14.44 s | 11.83 s |
| Peak process memory | 2.05 GiB | 1.64 GiB |
| Median latency on the repeated pair | 3.04 s | 3.29 s |

These are process measurements from a small smoke test while other development work could be active. Peak working set on Windows and peak RSS on Linux are related measurements with different OS accounting. They are not hardware minimums or a full-workload latency benchmark. The first pair includes cold query effects; the JSON evidence retains all measured latencies.

The largest observed score differences were 1.74e-06 for the Reader selection score and 1.19e-07 for the reranker. Scores need not be bitwise identical across numerical backends; the four paired decisions and answers were identical here.

The bundle manifest SHA-256 is `2834c057f845137c129dcbbd6bc419e0a63303779d0a2f31a0b7cff11a4a57b9`. The canonical freeze identity is `e2304cebdc5ed6af713f0a6a3376d15849394f882c718c81ce5ad4963d2b0574`. Its JSON file byte hash is recorded separately. The canonical freeze, source files, configuration and artifact binding were validated. No final holdout questions were read for this check.

Earlier separate launcher tests also exercised automatic Python preparation on Windows x64 and Linux x64 with Python absent from PATH. The launchers verified the pinned uv tool, installed Python 3.12.15 locally, installed a pure CPU environment and passed `pip check`. The final public download/bootstrap test is recorded separately after the metadata and descriptor are published.

The Linux environment ran through Docker on the same host solely for this verification. Users do not need Docker. This is a cross-environment smoke test, not external independent review or a claim that every input and hardware combination has been tested. Compact evidence is in [environment-verification.json](environment-verification.json); runtime instructions are in [runtime.md](../docs/runtime.md).
