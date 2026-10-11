# Bounded model training

[English](training.md) | [한국어](training.ko.md)

The training scripts turn the original notebook experiments into repeatable query-passage refinement and a matched random-negative versus hard-negative reranker comparison. Training uses official KorQuAD training questions only. The project development and final question sets do not enter training preparation, mining or checkpoint selection.

## Inputs and checkpoint identity

Prepare the runtime using the [runtime guide](runtime.md). Training is optional and substantially slower on CPU. The recorded experiment used PyTorch 2.14.0+cu130, Transformers 5.17.0 and an RTX 3060 Laptop GPU with 6 GB memory.

Place the original project checkpoints and their tokenizers in `.local/training-inputs/retriever` and `.local/training-inputs/reranker`. These inputs are the original trained models, before the additional refinement described here. A final runtime checkpoint is not an interchangeable training initializer. Compare the published artifact manifest and these identities before running:

| Input | SHA-256 |
| --- | --- |
| Original `week2_dense_retriever.pt` | `ec09ae94fbdf5e8e86e1766353bcb71ea49aa222f4096f4f2ea2bfb65e6c33ca` |
| Converted original retriever `model.safetensors` | `2ce1e42b5a20ca387b3ba81eafb90bdeb7c73907ee6b66ebf9b1c7e17f9d681b` |
| Original reranker `model.safetensors` | `98071cd9882265c172865279b91f75e0b40c6f5e9787be60296ee82fb5f29d2c` |
| Original KLUE architecture `config.json` | `71e0b0ea43d6b05c1a92484e0deb8a71b471f7787c3a7c97fc664e62347a08f5` |

The architecture configuration comes from `klue/roberta-base` revision `02f94ba5e3fcb7e2a58a390b8639b0fac974a8da`. It records 12 attention heads, 12 layers and a hidden size of 768. Conversion requires this explicit original configuration and loads weights strictly. Tensor shapes alone cannot recover the attention-head count. The original tokenizer identities and every initializer file are preserved in conversion and training manifests.

If the retriever is supplied as the legacy checkpoint rather than the converted Hugging Face directory, convert it once. Adjust the three source paths to the supplied archive layout:

```powershell
.venv\Scripts\python.exe -m scripts.train_models convert `
  --initial-model .local\training-inputs\week2_dense_retriever.pt `
  --tokenizer .local\training-inputs\tokenizer `
  --architecture-config .local\training-inputs\klue-roberta-base-config.json `
  --output-dir .local\training-inputs\retriever
```

All output directories below must be new. The scripts refuse to overwrite previous runs.

## Reconstruct and split the training data

First reconstruct the measured benchmark from unchanged official source downloads and the published ID recipe:

```powershell
.venv\Scripts\python.exe -m scripts.reproduce_data `
  --source-dir .local\source-data --output-dir .local\benchmark-v1

.venv\Scripts\python.exe -m scripts.prepare_training_data build `
  --train-source .local\benchmark-v1\train.jsonl `
  --tokenizer .local\training-inputs\retriever `
  --output-dir .local\training-data `
  --train-questions 3000 --dev-questions 200 `
  --chunk-max-tokens 256 --overlap-tokens 32 --seed 378
```

The benchmark recipe verifies the original download hashes, split counts and all five generated JSONL hashes. The training build reads `train.jsonl` only, removes duplicate normalized questions and selects whole paragraph groups with a fixed seed. Its internal development paragraphs have no overlap with the training paragraphs or training negative pool. Keeping groups intact produces 3,002 training questions from 478 paragraphs and 206 internal development questions from 34 paragraphs.

Natural chunks use sentence boundaries and token windows. Their boundaries do not depend on gold answer locations. Training labels select only chunks that contain the complete annotated answer span. The recorded data has 697 training chunks and 52 internal development chunks, with no selected question dropped for a missing complete span.

## Mine and match negative passages

```powershell
.venv\Scripts\python.exe -m scripts.prepare_training_data mine `
  --data-dir .local\training-data `
  --initial-model .local\training-inputs\retriever `
  --output-dir .local\mining `
  --device cuda --max-length 64 --top-k 50 --batch-size 16 --seed 378
```

Use `--device cpu` when CUDA is unavailable. Mining freezes the original project STS encoder, with masked mean pooling and L2 normalization. It searches the training and internal development corpora separately. Every negative excludes the gold paragraph, the same article title and any passage containing an annotated answer after text normalization. This removes known false negatives; incomplete answer annotations can still leave ambiguous passages.

Both arms retain the same question IDs, positive passages and example counts. If a valid random negative or hard negative is unavailable, the question is removed from both arms. The recorded run retained all 3,002 training and 206 internal development questions. `mining_manifest.json` and the preparation reports record inputs, output hashes, filtering and model identity.

## Train the bounded candidates

```powershell
.venv\Scripts\python.exe -m scripts.train_models train `
  --task retriever `
  --train .local\mining\train-hard-triplets.json `
  --dev .local\mining\dev-hard-triplets.json `
  --development-corpus .local\training-data\dev-corpus.jsonl `
  --initial-model .local\training-inputs\retriever `
  --config configs\training.json `
  --output-dir .local\runs\task-retriever-hard --device cuda

.venv\Scripts\python.exe -m scripts.train_models train `
  --task reranker `
  --train .local\mining\train-random-triplets.json `
  --dev .local\mining\dev-hard-triplets.json `
  --initial-model .local\training-inputs\reranker `
  --config configs\training.json `
  --output-dir .local\runs\reranker-random --device cuda

.venv\Scripts\python.exe -m scripts.train_models train `
  --task reranker `
  --train .local\mining\train-hard-triplets.json `
  --dev .local\mining\dev-hard-triplets.json `
  --initial-model .local\training-inputs\reranker `
  --config configs\training.json `
  --output-dir .local\runs\reranker-hard --device cuda
```

The two reranker runs initialize from exactly the same original project reranker. That initializer already underwent hard-negative training. This comparison measures additional random-negative versus hard-negative refinement, rather than two independent experiments starting from a pretrained language model. The historical reranker training subset was not preserved as question IDs, so its prior exposure to the new internal development pairs cannot be ruled out.

`configs/training.json` fixes seed 378, learning rate 1e-5, weight decay 0.01, at most two epochs, microbatch size 2 and accumulation over eight microbatches. The retriever uses query/passage limits of 128/256 tokens and temperature 0.05. The reranker uses a 384-token question-passage pair limit. AdamW, 10% linear warmup, gradient clipping, checkpointing and optional CUDA mixed precision keep memory use bounded. Accumulation weights reflect actual sample counts, including partial final batches.

The retriever loss compares each query with its positive and its explicit negative passage. It does not treat every other positive in a batch as a negative. Retriever checkpoint selection uses internal Recall@5 + 0.01 × MRR@20. Reranker selection uses the fraction of fixed internal development pairs for which the positive score exceeds the hard-negative score.

Before training, the initial model is evaluated and saved as the incumbent `best_model`. A checkpoint replaces it only when the declared development selection score improves by more than 0.001. The first non-improving epoch stops the run, with a maximum of two epochs. If no checkpoint improves, `selected_epoch` is zero and the initial weights remain selected. Each output includes `training_report.json`, `input_manifest.json` and a portable `best_model` directory.

## Recorded retriever result and promotion scope

The task-specific retriever selected epoch 1; epoch 2 degraded and triggered the development plateau stop. These measurements use 206 internal development questions against 52 chunks:

| Internal training metric | Initial model | Selected epoch 1 | Rejected epoch 2 |
| --- | ---: | ---: | ---: |
| Recall@1 | 84.95% | 93.20% | 91.75% |
| Recall@5 | 96.12% | 100.00% | 99.03% |
| Recall@20 | 99.03% | 100.00% | 100.00% |
| MRR@20 | 0.90275 | 0.96197 | 0.95449 |

The run took 790.85 seconds for 376 optimizer steps, with 2,232,149,504 bytes of peak PyTorch CUDA allocation. These retrieval and pair-ranking measurements are checkpoint-selection diagnostics. They are not final RAG answer accuracy and do not establish generalization to a large unseen corpus. Candidate promotion additionally requires the fixed project development RAG comparison. The final questions remain reserved until the model, reader and answer policy are frozen; see the published evaluation report for the selected runtime and final performance.

The matched reranker refinement experiment found no improvement under its declared selection rule:

| Additional refinement arm | Initial pair-ranking accuracy | Epoch 1 accuracy | Selected epoch | Optimizer steps | Run time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random negatives | 100.00% | 100.00% | 0 | 188 | 136.19 s |
| Hard negatives | 100.00% | 100.00% | 0 | 188 | 138.24 s |

Both runs stopped after epoch 1 and retained the original weights, verified by the identical `98071cd9...` SHA-256 listed above. The 100% initial accuracy is a ceiling on this small, fixed pair test. It does not show that the reranker always ranks a complete retrieval candidate list correctly or that either negative strategy is generally superior. No additional epoch or training variant was pursued after this plateau. Peak PyTorch CUDA allocation was 2,251,112,960 bytes for the random arm and 2,251,657,728 bytes for the hard arm.

Repeated training on a different device or software build can vary numerically. Input hashes and seeds make the experiment auditable; they do not guarantee identical floating-point checkpoints on every machine.
