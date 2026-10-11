# Public download verification

[English](public-download-verification.md) | [한국어](public-download-verification.ko.md)

The published source archive passed an actual clean Windows CPU bootstrap on 2026-10-11. The public `run.ps1 --doctor` created a new isolated environment, downloaded all three model weights and the metadata archive, fetched the original official KorQuAD development JSON and reconstructed the local corpus and chunks. Every declared file size and SHA-256 matched, the complete bundle manifest passed verification, and `pip check` and doctor both passed. No private model or dataset files, authenticated downloads or existing Hugging Face model cache were used.

| Measured step | Result |
| --- | --- |
| First public bootstrap and doctor | 223.43 s |
| First persistent CLI session | 2 questions, 20.95 s |
| Prepared `--no-download` persistent session | 4 questions, 33.55 s |
| Total demo inferences | 6, from 2 distinct public authored examples |
| Model loading | Once per persistent session |
| Repeated answer, decision and citation identities | All matched across repeats and sessions |
| Python / PyTorch | 3.12.7 / 2.14.0+cpu |
| System packages shared with the venv | No |

Both examples returned `abstained`, consistently across the two sessions. This confirms execution and the abstention response path, rather than correctness on an answerable benchmark. Both sessions used the actual public launcher in JSON mode and ended successfully with `/quit`. The reuse session used `--no-download`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and local-only model loading under the restricted shell sandbox. The Hugging Face cache remained empty. No separate packet-capture or network-firewall claim is made.

The default launcher reused an existing Python 3.12 interpreter for this run; automatic installation when Python is absent from PATH was exercised separately on Windows x64 and Linux x64. The installer downloaded dependencies into a new venv with `--no-cache-dir`. These timings include the network conditions and machine load of this one run and are not expected installation times for every user.

The actual public source archive used here was 618,451 bytes, SHA-256 `88240025f30e35fb3e1a5928b4821f0defc7cb3e239120281cd71cfc260c8b02`. This identifies the initial published archive tested, rather than every later documentation-only revision of the archive. All 19 frozen core files remained unchanged after installation and both sessions. Their identities, the exact public URLs and the version inventory are preserved in [public-download-verification.json](public-download-verification.json).

| Public artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Runtime metadata ZIP | 7,580,697 | `73537c36b5f33743b150db5fc7fcff91fb65cd2174693290fed75a3808f06d8f` |
| Reader weight, pinned official Hugging Face revision | 449,357,300 | `75ad3ca646cd32eefbc8f34cb17b34f191783edcf537fd2f3f865f25617240ce` |
| Original reranker weight, existing public Drive file | 442,502,720 | `98071cd9882265c172865279b91f75e0b40c6f5e9787be60296ee82fb5f29d2c` |
| E5 retriever weight, pinned official Hugging Face revision | 470,641,600 | `1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477` |
| Unmodified official KorQuAD development JSON | 3,881,058 | `25ffeb51e6c51ec02c071b60a10188e10005c144110f0d876b26079d80a35bdf` |

These artifact downloads total 1,373,963,375 bytes, excluding source and dependencies. The metadata archive carries reconstruction offsets and identities, with no redistributed paragraph text. The installer derives the local corpus and chunk files from the verified official source before checking the complete manifest. The resulting manifest SHA-256 is `2834c057f845137c129dcbbd6bc419e0a63303779d0a2f31a0b7cff11a4a57b9`, bound to canonical freeze `e2304cebdc5ed6af713f0a6a3376d15849394f882c718c81ce5ad4963d2b0574`.

This is a fresh-source and public-delivery check on the same host, not external independent review. The separate Windows/Linux CPU portability smoke and final synthetic CPU audit retain their own scopes in [environment-verification.md](environment-verification.md) and [synthetic-final-cpu-audit.md](synthetic-final-cpu-audit.md). Runtime instructions are in [runtime.md](../docs/runtime.md).
