# Runtime and reproduction

[English](runtime.md) | [한국어](runtime.ko.md)

The public launchers prepare a project-local Python environment, verify the recorded model bundle and start a persistent Korean question-answering session. CPU execution is the default. No Docker, hosted LLM, API key or global package installation is required.

## Start from the source

Clone the repository or extract the [portfolio source archive](https://park-dongmyung.com/downloads/korean-nlp-rag-source.zip) into a writable directory. Keep its directory structure intact. Run the launcher from that directory:

```powershell
# Windows PowerShell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1
```

Windows users can also double-click `launch-rag.bat`. On Linux:

```sh
sh run.sh
```

The first run needs internet access to download the pinned Python runtime when necessary, pinned dependencies, model metadata and model weights. An existing 64-bit Python 3.12 is reused when available. Otherwise, the supported Windows x64 and Linux x64 launchers verify a pinned official uv download and install Python 3.12.15 beneath `.local/`. Python downloads, uv caches and the `.venv` remain inside this project. Linux needs `curl` or `wget`, `tar` and `sha256sum` only when this bootstrap is necessary.

Enter one question at a time. `/quit` or end-of-input ends the session. The models are loaded once and reused. The fixed source corpus cannot answer every question; an abstention is a supported result.

## Single questions, batches and CUDA

```powershell
.\run.ps1 --question "직지심체요절은 언제 인쇄되었나요?" --json
.\run.ps1 --questions-file .\questions.txt --json
.\run.ps1 --no-download --interactive
```

`questions.txt` must contain one UTF-8 question per line. Batch output contains one JSON object per question. Linux accepts the same arguments after `sh run.sh`.

CUDA is optional and must be requested explicitly:

```powershell
.\run.ps1 --device cuda --question "직지심체요절은 언제 인쇄되었나요?" --json
```

This prepares the pinned CUDA 13.0 PyTorch distribution and checks that a compatible NVIDIA device and driver are available. Changing the runtime distribution can cause an existing incompatible project environment to be moved into `.local/environment-backups/` before a clean replacement is created. The launcher does not install or change GPU drivers. GPU benchmark results should be interpreted with the recorded hardware and environment metadata.

## Setup and diagnostics

```powershell
# Prepare dependencies without downloading or loading models
.\run.ps1 --setup-only

# Prepare the model bundle and verify installed versions and every artifact
.\run.ps1 --doctor

# Verify the existing bundle without downloading anything
.venv\Scripts\python.exe scripts\download_artifacts.py --destination artifacts\runtime --verify-only

# Run the CLI directly after setup
.venv\Scripts\python.exe -m korean_rag --interactive --bundle artifacts\runtime --device cpu
```

Equivalent Linux commands use `.venv/bin/python`. Standalone setup is available as `python scripts/setup_runtime.py --device cpu`; use `--repair` to preserve and replace an incompatible project environment. The download command is `python scripts/download_artifacts.py --manifest reports/artifact-download.json --destination artifacts/runtime`. Both scripts also support module execution, for example `python -m scripts.doctor --bundle artifacts/runtime`.

`--doctor` reports Python and pinned package versions, artifact validity and problems as JSON. It checks the files but does not measure answer quality. Missing or altered files are rejected before model loading. An incomplete artifact destination is preserved rather than silently treated as ready or overwritten; choose a new `--bundle` directory after investigating the error. A valid prepared environment and bundle can be reused offline with `--no-download`.

## Artifact provenance and local source reconstruction

`reports/artifact-download.json` records public download URLs, exact byte sizes and SHA-256 hashes. A small metadata ZIP is served from the portfolio. Unmodified public-model weights are downloaded from their recorded official Hugging Face revisions, and project-trained weights use the recorded public artifact link. This avoids depending on authenticated GitHub access or republishing upstream weights. Archives are checked before extraction, and each separate model download is verified before the bundle becomes usable. Extraction rejects traversal paths, filesystem links, unsafe file types and duplicate filenames; separate files cannot replace archive entries. The final manifest binds the complete model, tokenizer, index, configuration and source-file inventory.

The metadata archive omits model weights, `corpus.jsonl` and `chunks.jsonl`. The descriptor's `data_recipe` downloads the unchanged [official KorQuAD validation source](https://korquad.github.io/dataset/KorQuAD_v1.0_dev.json), verifies its recorded size and hash, and reconstructs the source corpus and chunks locally from paragraph IDs and character offsets. It then verifies those exact generated bytes against the bundle manifest. The published recipe contains source identities and offsets rather than copied paragraph or question text. KorQuAD is distributed under CC BY-ND 2.0 KR; see the dataset credits and license notices in this repository.

The runtime loads local Hugging Face `safetensors` weights without model-code downloads or pickle checkpoints. It rebuilds BM25 from the verified source chunks. Source byte identities are preserved across Windows and Linux. Identical artifacts do not guarantee bitwise-identical floating-point computation on every processor or GPU.

## Reproduce the benchmark data and evaluate

After setup, reconstruct the measured split from the recorded official downloads:

```powershell
.venv\Scripts\python.exe scripts\reproduce_data.py --source-dir .local\source-data --output-dir .local\benchmark-v1
```

The output directory must be new. The recipe checks split separation, measured counts and exact file hashes. Use `--offline` when the verified original downloads are already in the source directory.

For development evaluation:

```powershell
.venv\Scripts\python.exe scripts\evaluate_portable.py --bundle artifacts\runtime --benchmark .local\benchmark-v1 --split dev --device cpu --output .local\evaluation-dev
```

The evaluation output directory must also be new. It records predictions, metrics, source and artifact hashes, environment metadata and measured loading and inference times. Development evaluation can be expensive on CPU. Final evaluation additionally requires the published freeze record via `--freeze`; its bound source files, configuration and input hashes must match before final questions are read. Follow the evaluation report for the exact final inputs, including the separately prepared no-answer suite when specified. An arbitrary new question list cannot reproduce the published benchmark score.

### Explicit preparation of the original project retriever

The selected runtime does not use `.pt` checkpoints. To reproduce the comparison with the original development model, manually download the [original project checkpoint](https://drive.google.com/file/d/1kairfruRDX3xTYU3u_8h4VdZwoMyE887/view) as `.local/original/week2_dense_retriever.pt`. Download the [original metadata archive](https://park-dongmyung.com/downloads/korean-rag-original-metadata.zip) and extract it into `.local/original-metadata`. Its `models/retriever` directory must contain only `config.json`, `tokenizer.json` and `tokenizer_config.json`. Run this offline preparation command with the installed project environment:

```powershell
.venv\Scripts\python.exe scripts\convert_original_retriever.py --checkpoint .local\original\week2_dense_retriever.pt --metadata .local\original-metadata\models\retriever --output .local\original-retriever
```

The checkpoint SHA-256 must be `ec09ae94fbdf5e8e86e1766353bcb71ea49aa222f4096f4f2ea2bfb65e6c33ca`. The script verifies that checkpoint and all metadata before calling the CPU `weights_only=True` loader. It refuses arbitrary checkpoints, downloads nothing and preserves existing output. It publishes the portable directory only if `model.safetensors` matches the measured original model's exact hash `2ce1e42b5a20ca387b3ba81eafb90bdeb7c73907ee6b66ebf9b1c7e17f9d681b` and size 442,494,816 bytes. The conversion report is written beside the model directory. This conversion was actually completed in a separate pure CPU environment and produced the exact expected bytes. It prepares model weights; it does not build an original comparison index or reproduce the benchmark score by itself.

Prepare the original comparison bundle by supplying that exact converted file:

```powershell
.venv\Scripts\python.exe scripts\download_artifacts.py --manifest reports\original-artifact-download.json --destination artifacts\original --local-retriever .local\original-retriever\model.safetensors
```

This command downloads the remaining recorded metadata and public weights, reconstructs source data locally and copies the converted retriever only into its declared `models/retriever/model.safetensors` path. Both its SHA-256 and byte size must match the original bundle manifest. A conflicting downloaded model, changed local file or existing destination is rejected without overwriting it. After the original bundle is ready, the original comparison's final evaluation uses `configs/original-freeze.json`; the selected model uses its own freeze record. Follow the evaluation report for the exact no-answer inputs and output paths.

## Resource measurements and validation scope

Allow at least 6 GB of free disk space for a practical CPU installation budget, including models, the virtual environment and temporary downloads. This is a planning allowance rather than a measured universal minimum. The selected final bundle occupies about 1.39 GB and the clean Windows CPU virtual environment about 0.98 GB. CUDA dependencies, retained environment backups, optional original-model comparison artifacts and benchmark outputs require additional space. 8 GB or more of system RAM provides useful headroom; it is not a validated lower bound for every workload.

Final offline tests used two predeclared development questions, each repeated twice, in separate clean Windows and Linux CPU environments on the same computer. Their frozen source and artifact hashes, answers, decisions and citation identities matched. Peak process memory was 2.05 GiB on Windows and 1.64 GiB on Linux, with four CPU threads. Windows and Linux automatic Python bootstrap and CPU dependency installation were also executed successfully in separate initial preparation tests. These small inference tests establish the measured portability scope, not the full final benchmark or external independent review. Details and numerical differences are recorded in [environment-verification.md](../reports/environment-verification.md).
