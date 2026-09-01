"""End-to-end local inference pipeline for the Week 4 Korean RAG system."""

from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass

import numpy as np

from .reranking import CrossEncoderReranker
from .retrieval import ArtifactPaths, load_week2_retriever, reciprocal_rank_fusion, tokenize_for_bm25, top_k_from_scores

DEFAULT_READER_MODEL = "monologg/koelectra-base-v3-finetuned-korquad"
DEFAULT_MIN_RRF_SCORE = 0.025
DEFAULT_MIN_RERANKER_SCORE = 0.50
DEFAULT_MIN_READER_SCORE = 0.10


@dataclass
class AnswerCandidate:
    answer: str
    score: float
    rank: int
    chunk_id: int
    title: str
    context: str


class KoreanRAGPipeline:
    """BM25 + dense retrieval + RRF + reranking + extractive QA."""

    def __init__(self, artifacts: ArtifactPaths, device: str = "auto", reader_model: str = DEFAULT_READER_MODEL, use_reader: bool = True) -> None:
        import faiss
        import pandas as pd
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers.utils import logging as transformers_logging

        transformers_logging.disable_progress_bar()
        artifacts.validate()
        self.artifacts = artifacts
        self.device = self._resolve_device(device, torch)
        self.chunks = pd.read_csv(artifacts.week4_dir / "week4_knowledge_base_chunks.csv")
        self.dense_index = faiss.read_index(str(artifacts.week4_dir / "week4_dense.index"))
        with (artifacts.week4_dir / "week4_bm25.pkl").open("rb") as file:
            self.bm25 = pickle.load(file)
        self.retriever = load_week2_retriever(artifacts.week2_dir / "week2_dense_retriever.pt", artifacts.week2_dir / "tokenizer", self.device)
        reranker_model = AutoModelForSequenceClassification.from_pretrained(artifacts.week3_dir, local_files_only=True)
        reranker_tokenizer = AutoTokenizer.from_pretrained(artifacts.week3_dir, local_files_only=True)
        self.reranker = CrossEncoderReranker(reranker_model, reranker_tokenizer, self.device, batch_size=32, max_length=512)
        self.reader = self._load_reader(reader_model) if use_reader else None

    @staticmethod
    def _resolve_device(device: str, torch):
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")
        return torch.device(device)

    def _load_reader(self, reader_model: str):
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(reader_model, local_files_only=True)
            model = AutoModelForQuestionAnswering.from_pretrained(
                reader_model, local_files_only=True
            )
        except OSError:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="You are sending unauthenticated requests to the HF Hub.*",
                )
                tokenizer = AutoTokenizer.from_pretrained(reader_model)
                model = AutoModelForQuestionAnswering.from_pretrained(reader_model)
        model = model.to(self.device)
        model.eval()
        return tokenizer, model

    def _read_span(self, question: str, context: str) -> tuple[str, float]:
        """Extract the most likely answer span from a single evidence chunk."""
        import torch

        tokenizer, model = self.reader
        encoded = tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=512,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        sequence_ids = encoded.sequence_ids(0)
        model_inputs = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = model(**model_inputs)
        start_probabilities = torch.softmax(outputs.start_logits[0], dim=-1).cpu().numpy()
        end_probabilities = torch.softmax(outputs.end_logits[0], dim=-1).cpu().numpy()
        context_tokens = [
            index for index, sequence_id in enumerate(sequence_ids)
            if sequence_id == 1 and offsets[index][1] > offsets[index][0]
        ]
        best = ("", 0.0)
        for start in context_tokens:
            for end in context_tokens:
                if end < start or end - start + 1 > 40:
                    continue
                start_offset, end_offset = offsets[start][0], offsets[end][1]
                score = float(start_probabilities[start] * end_probabilities[end])
                if score > best[1]:
                    best = (context[start_offset:end_offset].strip(), score)
        return best

    @staticmethod
    def _answer_is_supported(answer: str, context: str) -> bool:
        """Confirm that the accepted answer is present in its cited evidence."""
        normalized_answer = "".join(answer.lower().split())
        normalized_context = "".join(context.lower().split())
        return bool(normalized_answer) and normalized_answer in normalized_context

    @staticmethod
    def _abstain(evidence: list[dict], reasons: list[str]) -> dict:
        return {
            "answer": "현재 지식베이스의 근거만으로는 답을 확인할 수 없습니다.",
            "answer_source": "abstained",
            "abstention_reasons": reasons,
            "evidence": evidence,
        }

    def search(self, question: str, candidate_k: int = 50, rerank_k: int = 20) -> list[dict]:
        """Retrieve and rerank evidence chunks for one Korean question."""
        if not question or not question.strip():
            raise ValueError("Question must not be empty.")
        bm25_scores = np.asarray(self.bm25.get_scores(tokenize_for_bm25(question)))
        bm25_indices, _ = top_k_from_scores(bm25_scores, candidate_k)
        _, dense_indices = self.dense_index.search(self.retriever.encode([question]), min(candidate_k, len(self.chunks)))
        fused = reciprocal_rank_fusion(bm25_indices, dense_indices[0])[:rerank_k]
        chunk_ids = [chunk_id for chunk_id, _ in fused]
        contexts = [str(self.chunks.iloc[chunk_id]["chunk_text"]) for chunk_id in chunk_ids]
        reranker_scores = self.reranker.score([question] * len(contexts), contexts)
        records = []
        for chunk_id, (_, rrf_score), score in zip(chunk_ids, fused, reranker_scores):
            row = self.chunks.iloc[chunk_id]
            records.append({"chunk_id": int(chunk_id), "title": str(row["title"]), "context": str(row["chunk_text"]), "reranker_score": float(score), "rrf_score": float(rrf_score)})
        records.sort(key=lambda item: item["reranker_score"], reverse=True)
        for rank, record in enumerate(records, start=1):
            record["rank"] = rank
        return records

    def answer(
        self,
        question: str,
        context_k: int = 3,
        min_rrf_score: float = DEFAULT_MIN_RRF_SCORE,
        min_reranker_score: float = DEFAULT_MIN_RERANKER_SCORE,
        min_reader_score: float = DEFAULT_MIN_READER_SCORE,
    ) -> dict:
        """Answer from the highest-confidence span in reranked evidence."""
        evidence = self.search(question)[:context_k]
        if self.reader is None:
            return {"answer": None, "answer_source": "retrieval_only", "evidence": evidence}

        retrieval_reasons = []
        if not evidence:
            retrieval_reasons.append("No evidence was retrieved.")
        elif evidence[0]["rrf_score"] < min_rrf_score:
            retrieval_reasons.append(f"Hybrid retrieval confidence was below {min_rrf_score:.3f}.")
        elif evidence[0]["reranker_score"] < min_reranker_score:
            retrieval_reasons.append(f"Reranker relevance was below {min_reranker_score:.2f}.")
        if retrieval_reasons:
            return self._abstain(evidence, retrieval_reasons)
        candidates: list[AnswerCandidate] = []
        for item in evidence:
            answer, score = self._read_span(question, item["context"])
            if len(answer.replace(" ", "")) >= 2:
                candidates.append(AnswerCandidate(answer, score, item["rank"], item["chunk_id"], item["title"], item["context"]))
        if not candidates:
            return self._abstain(evidence, ["The QA reader did not produce a valid answer span."])
        best = max(candidates, key=lambda candidate: candidate.score)
        if best.score < min_reader_score:
            return self._abstain(evidence, [f"Reader confidence was below {min_reader_score:.2f}."])
        if not self._answer_is_supported(best.answer, best.context):
            return self._abstain(evidence, ["The extracted answer was not found in its cited evidence."])
        return {
            "answer": best.answer,
            "answer_source": "extractive_reader",
            "reader_score": best.score,
            "answer_supported": True,
            "citation": {"title": best.title, "chunk_id": best.chunk_id, "context": best.context},
            "evidence": evidence,
        }
