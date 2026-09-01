"""Local retrieval utilities for the Korean RAG portfolio."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


def mean_pooling(model_output, attention_mask):
    """Create one embedding per sequence using masked mean pooling."""
    import torch

    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


class RobertaDenseRetriever:
    """Week 2 KLUE-RoBERTa encoder restored from its saved checkpoint."""

    def __init__(self, model, tokenizer, device, max_length: int) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.model.eval()

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        import torch

        embeddings: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            encoded = self.tokenizer(
                list(texts[start : start + batch_size]), padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                pooled = mean_pooling(self.model(**encoded), encoded["attention_mask"])
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(embeddings, axis=0)


def _local_roberta_config(state_dict):
    """Reconstruct KLUE-RoBERTa's architecture without downloading its base model."""
    from transformers import RobertaConfig

    return RobertaConfig(
        vocab_size=state_dict["roberta.embeddings.word_embeddings.weight"].shape[0],
        hidden_size=state_dict["roberta.embeddings.word_embeddings.weight"].shape[1],
        num_hidden_layers=len({key.split(".")[3] for key in state_dict if key.startswith("roberta.encoder.layer.")}),
        num_attention_heads=12,
        intermediate_size=state_dict["roberta.encoder.layer.0.intermediate.dense.weight"].shape[0],
        max_position_embeddings=state_dict["roberta.embeddings.position_embeddings.weight"].shape[0],
        type_vocab_size=state_dict["roberta.embeddings.token_type_embeddings.weight"].shape[0],
        pad_token_id=1, bos_token_id=0, eos_token_id=2, layer_norm_eps=1e-5,
    )


def load_week2_retriever(checkpoint_path: Path, tokenizer_path: Path, device) -> RobertaDenseRetriever:
    """Load Week 2's dense retriever from the compact local artifact set."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    saved_state = checkpoint["model_state_dict"]
    model = AutoModel.from_config(_local_roberta_config(saved_state))
    model.load_state_dict(
        {key.removeprefix("roberta."): value for key, value in saved_state.items()}
    )
    return RobertaDenseRetriever(model, tokenizer, device, int(checkpoint["max_length"]))


def tokenize_for_bm25(text: str) -> list[str]:
    """Use the same lightweight Korean tokenization used in Week 4."""
    import re

    normalized = " ".join(str(text).split()).lower()
    tokens = re.findall(r"[가-힣A-Za-z0-9]+", normalized)
    return tokens if tokens else [normalized]


def top_k_from_scores(scores: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return descending top-k indices and scores."""
    k = min(top_k, len(scores))
    candidate_indices = np.argpartition(scores, -k)[-k:]
    order = np.argsort(scores[candidate_indices])[::-1]
    indices = candidate_indices[order].astype(np.int64)
    return indices, scores[indices].astype(np.float32)


def reciprocal_rank_fusion(sparse_indices: Sequence[int], dense_indices: Sequence[int], rrf_k: int = 60) -> list[tuple[int, float]]:
    """Fuse BM25 and dense rankings using the Week 4 RRF rule."""
    scores: dict[int, float] = {}
    for ranking in (sparse_indices, dense_indices):
        for rank, chunk_id in enumerate(ranking, start=1):
            chunk_id = int(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


@dataclass(frozen=True)
class ArtifactPaths:
    """Locations of the persisted Week 2-4 artifacts."""

    outputs_root: Path

    @property
    def week2_dir(self) -> Path:
        return self.outputs_root / "week2" / "week2_retriever_artifacts"

    @property
    def week3_dir(self) -> Path:
        return self.outputs_root / "week3" / "week3_day4_final_reranker"

    @property
    def week4_dir(self) -> Path:
        return self.outputs_root / "week4" / "week4_day1_3_artifacts"

    def validate(self) -> None:
        required = [
            self.week2_dir / "week2_dense_retriever.pt", self.week2_dir / "tokenizer",
            self.week3_dir / "config.json", self.week3_dir / "model.safetensors",
            self.week4_dir / "week4_knowledge_base_chunks.csv", self.week4_dir / "week4_dense.index",
            self.week4_dir / "week4_bm25.pkl",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Required artifacts are missing:\n- " + "\n- ".join(missing))

    def retrieval_config(self) -> dict:
        return json.loads((self.week2_dir / "retrieval_config.json").read_text(encoding="utf-8"))
