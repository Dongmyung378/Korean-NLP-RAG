"""Inference utilities for Cross-Encoder reranking."""

from __future__ import annotations

from collections.abc import Sequence


class CrossEncoderReranker:
    """Batch score question-context pairs with a binary relevance classifier."""

    def __init__(
        self,
        model,
        tokenizer,
        device,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.model.eval()

    def score(self, questions: Sequence[str], contexts: Sequence[str]) -> list[float]:
        if len(questions) != len(contexts):
            raise ValueError("questions and contexts must have the same length")

        # Keep this import local so metrics can be used in environments that
        # only need NumPy (for example, lightweight result-reporting scripts).
        import torch

        scores: list[float] = []
        for start in range(0, len(questions), self.batch_size):
            batch = self.tokenizer(
                list(questions[start : start + self.batch_size]),
                list(contexts[start : start + self.batch_size]),
                padding=True,
                truncation="only_second",
                max_length=self.max_length,
                return_tensors="pt",
            )
            with torch.inference_mode():
                logits = self.model(
                    **{key: value.to(self.device) for key, value in batch.items()}
                ).logits
            scores.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
        return scores
