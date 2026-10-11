# Public unit-suite verification

[English](verification-tests.md) | [한국어](verification-tests.ko.md) | [Recorded results](verification-tests.json)

The public suite passed in separate Windows and Linux CPU environments. Each full run executed 112 tests: 109 passed, three optional historical-artifact tests were skipped, and none failed. These are code checks rather than a model accuracy benchmark.

| Environment | Python | PyTorch | Tests passed | Historical skips | Failures | Unit-suite time |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Separate Windows CPU virtual environment | 3.12.7 | 2.14.0+cpu | 109 | 3 | 0 | 16.044 s |
| Isolated Linux CPU virtual environment | 3.12.15 | 2.14.0+cpu | 109 | 3 | 0 | 4.165 s |

Both environments exclude system site packages. The Linux run used Transformers 5.17.0, four CPU threads, an existing native Docker volume and a temporary `python:3.12-slim` container on the same host. Networking was disabled, Hugging Face offline mode was enabled, CUDA was unavailable, and the current source tree was mounted read-only. The temporary container was removed after the run; the prepared environment volume was retained.

Two tests previously placed scratch directories beneath the source tree. Their scratch directories now use the operating system's writable temporary directory, which permits verification from a read-only checkout. Their assertions were preserved. All 19 frozen core source files were checked against the recorded hashes and remained unchanged. The Windows full-run record predates this test-only portability adjustment; the Linux full run verifies the adjusted files.

The first Linux run passed in 5.267 seconds. A second run captured the private full stdout log and passed in 4.165 seconds. The JSON record distinguishes those runs and preserves the log SHA-256 without publishing local machine paths. Run times include the tiny random-model CPU training smoke test, which performs one optimizer step. No downloaded production model or full dataset inference is required by this suite.

The three skips concern historical Week 4 saved-result artifacts that are deliberately outside the public source tree. Skipped checks are not counted as passes. This same-host isolated verification is separate from [hosted CI status](ci-status.md): GitHub rejected workflow dispatch with HTTP 422 because Actions was disabled for the account, so no hosted CI pass is claimed.

After runtime setup, the public command is:

```sh
python -m unittest discover -s tests -v
```
