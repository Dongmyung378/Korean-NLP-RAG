# Optional training-model archive

[English](training-artifacts.md) | [한국어](training-artifacts.ko.md) | [Recorded size, SHA-256 and payload inventory](training-artifacts.json)

`korean-rag-training-models.zip` is prepared for manual publication. Its download link is pending; no public URL has been assigned. The archive is optional and is not required by the question-answering launcher.

It contains the exact original project STS retriever converted to Hugging Face format, the epoch-1 task-trained retriever, sanitized training and input provenance, the bounded training configuration, English and Korean instructions, and upstream KLUE attribution and license notices. Reader weights, raw KorQuAD questions and paragraphs, mined text pairs and personal records are excluded. Every payload file has a recorded SHA-256, and the release was checked by full ZIP CRC and payload-hash verification.

## Publish the prepared file

1. Upload the unchanged `korean-rag-training-models.zip` release file to Google Drive. Use the exact filename and keep the recorded size and hash from the JSON manifest.
2. Open its sharing settings, set general access to **Anyone with the link**, and set the role to **Viewer**. Copy the actual share link.
3. Test the link in a signed-out or private browser session. Download the complete ZIP and compare its byte size and SHA-256 with the manifest. A preview page or a link that still requires account approval is not a verified public download.
4. After the anonymous download passes, publish that real link in this report and in `training-artifacts.json`. The current `download_url` and `drive_file_id` remain `null` until a real upload has been verified.

This release has not been uploaded by the coding agent. The source repository's launcher uses its separate runtime descriptor; uploading these optional training models does not change the selected runtime.

## Verify and use a downloaded copy

```powershell
Get-FileHash -LiteralPath .\korean-rag-training-models.zip -Algorithm SHA256
(Get-Item -LiteralPath .\korean-rag-training-models.zip).Length
```

Compare both values with [the manifest](training-artifacts.json). Extract to a new writable directory and follow the archive's `README.md` or `README.ko.md`. Copy `models/original-retriever` to the source repository's `.local/training-inputs/retriever`, then follow [the training guide](../docs/training.md). The original reranker is a separate input; the verified selected runtime retains its original weights and can supply that initializer. The task-trained checkpoint is an experiment candidate and must not replace the original initializer when replaying the recorded training experiment.

The archived training reports describe internal checkpoint selection, not final RAG accuracy. The original and task-trained model identities and the known limits of the matched reranker comparison remain explicit in the accompanying documentation.
