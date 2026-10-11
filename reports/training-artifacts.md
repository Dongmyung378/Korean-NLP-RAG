# Optional training-model archive

[English](training-artifacts.md) | [한국어](training-artifacts.ko.md) | [Recorded size, SHA-256 and payload inventory](training-artifacts.json)

The optional `korean-rag-training-models.zip` is publicly available from [Google Drive](https://drive.google.com/file/d/1attqOUViSx2i4q0JcRz2l577HZTGIuh6/view?usp=drivesdk). A [direct download](https://drive.usercontent.google.com/download?id=1attqOUViSx2i4q0JcRz2l577HZTGIuh6&export=download&confirm=t) was verified without an account on October 11, 2026. The archive is not required by the question-answering launcher.

It contains the exact original project STS retriever converted to Hugging Face format, the epoch-1 task-trained retriever, sanitized training and input provenance, the bounded training configuration, English and Korean instructions, and upstream KLUE attribution and license notices. Reader weights, raw KorQuAD questions and paragraphs, mined text pairs and personal records are excluded. Every payload file has a recorded SHA-256, and the release was checked by full ZIP CRC and payload-hash verification.

## Published download verification

Google Drive sharing is **Anyone with the link, Viewer**. The complete public response returned HTTP 200 with `application/octet-stream`, exactly **790,806,262 bytes** and SHA-256 `9cbb8be9921dcf96a15f95fcc5c55f075be1344ceb3a2b6836b8bf366ba72677`. Verification used no credentials, authorization headers or cookies. These values match the prepared release; its recorded payload inventory is unchanged. The verification timestamp and download identity are in [the manifest](training-artifacts.json).

The final question-answering runtime uses multilingual E5 and its separate runtime descriptor. This archive preserves the original STS retriever and a task-trained candidate for training inspection and reproduction; adding it to the existing Drive folder does not replace the final runtime models or change the published evaluation results.

## Verify and use a downloaded copy

```powershell
Get-FileHash -LiteralPath .\korean-rag-training-models.zip -Algorithm SHA256
(Get-Item -LiteralPath .\korean-rag-training-models.zip).Length
```

Compare both values with [the manifest](training-artifacts.json). Extract to a new writable directory and follow the archive's `README.md` or `README.ko.md`. Copy `models/original-retriever` to the source repository's `.local/training-inputs/retriever`, then follow [the training guide](../docs/training.md). The original reranker is a separate input; the verified selected runtime retains its original weights and can supply that initializer. The task-trained checkpoint is an experiment candidate and must not replace the original initializer when replaying the recorded training experiment.

The archived training reports describe internal checkpoint selection, not final RAG accuracy. The original and task-trained model identities and the known limits of the matched reranker comparison remain explicit in the accompanying documentation.
