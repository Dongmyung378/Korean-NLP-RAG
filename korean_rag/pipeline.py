"""End-to-end local inference pipeline for the Week 4 Korean RAG system."""

from __future__ import annotations

import pickle
import re
import warnings
from dataclasses import dataclass

import numpy as np

from .reranking import CrossEncoderReranker
from .validation import validate_question_tokens
from .retrieval import ArtifactPaths, load_week2_retriever, reciprocal_rank_fusion, tokenize_for_bm25, top_k_from_scores

DEFAULT_READER_MODEL = "monologg/koelectra-base-v3-finetuned-korquad"
DEFAULT_MIN_RRF_SCORE = 0.025
DEFAULT_MIN_RERANKER_SCORE = 0.50
DEFAULT_MIN_READER_SCORE = 0.10


def decode_answer_span(context, offsets, sequence_ids, start_logits, end_logits, cls_index, max_answer_length=40):
    """문맥 토큰과 CLS만 비교하고, 빈 답보다 점수가 높은 구간을 반환한다."""
    context_tokens = [
        i for i, sequence_id in enumerate(sequence_ids)
        if sequence_id == 1 and offsets[i][1] > offsets[i][0]
    ]
    if not context_tokens:
        return "", 0.0
    valid_tokens = [cls_index, *context_tokens]
    probabilities = []
    for logits in (start_logits, end_logits):
        values = np.asarray(logits, dtype=np.float64)
        masked = np.full_like(values, -np.inf)
        masked[valid_tokens] = values[valid_tokens]
        weights = np.exp(masked - np.max(masked))
        probabilities.append(weights / weights.sum())
    starts, ends = probabilities
    null_score = float(starts[cls_index] * ends[cls_index])
    best = ("", null_score)
    for start in context_tokens:
        for end in context_tokens:
            if end < start or end - start + 1 > max_answer_length:
                continue
            score = float(starts[start] * ends[end])
            if score > best[1]:
                answer = context[offsets[start][0]:offsets[end][1]].strip()
                if answer:
                    best = (answer, score)
    return best


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
        self._validate_index_alignment()
        reranker_model = AutoModelForSequenceClassification.from_pretrained(artifacts.week3_dir, local_files_only=True)
        reranker_tokenizer = AutoTokenizer.from_pretrained(artifacts.week3_dir, local_files_only=True)
        self.reranker = CrossEncoderReranker(reranker_model, reranker_tokenizer, self.device, batch_size=32, max_length=512)
        self.reader = self._load_reader(reader_model) if use_reader else None

    def _validate_index_alignment(self) -> None:
        required = {"chunk_id", "title", "chunk_text"}
        if not required.issubset(self.chunks.columns):
            raise ValueError(f"Knowledge base must contain columns: {sorted(required)}")
        count = len(self.chunks)
        if not count or self.dense_index.ntotal != count or self.bm25.corpus_size != count:
            raise ValueError("Knowledge base, FAISS index and BM25 corpus sizes do not match.")
        if not np.array_equal(self.chunks["chunk_id"].to_numpy(), np.arange(count)):
            raise ValueError("Knowledge base chunk IDs must match row order, starting at zero.")
        if self.dense_index.d != self.retriever.model.config.hidden_size:
            raise ValueError("FAISS dimension does not match the dense retriever.")

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
        question = validate_question_tokens(question, tokenizer)
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
        token_ids = encoded["input_ids"][0].tolist()
        if tokenizer.cls_token_id not in token_ids:
            raise ValueError("The QA reader tokenizer must provide a CLS token.")
        cls_index = token_ids.index(tokenizer.cls_token_id)
        model_inputs = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = model(**model_inputs)
        return decode_answer_span(
            context, offsets, sequence_ids,
            outputs.start_logits[0].float().cpu().numpy(),
            outputs.end_logits[0].float().cpu().numpy(), cls_index,
        )

    @staticmethod
    def _answer_in_evidence(answer: str, context: str) -> bool:
        """Confirm that the accepted answer is present in its cited evidence."""
        normalized_answer = "".join(answer.lower().split())
        normalized_context = "".join(context.lower().split())
        return bool(normalized_answer) and normalized_answer in normalized_context

    @staticmethod
    def _explicit_years_match(question: str, context: str) -> bool:
        # 질문에 명시된 연도가 근거에 없으면 답변을 보류한다. 의미적 검증은 아니다.
        years = set(re.findall(r"(?<!\d)(\d{4})\s*년", question))
        context_years = set(re.findall(r"(?<!\d)(\d{4})\s*년", context))
        return years.issubset(context_years)

    @staticmethod
    def validate_answer_options(context_k, min_rrf_score, min_reranker_score, min_reader_score):
        if not isinstance(context_k, int) or not 1 <= context_k <= 20:
            raise ValueError("context_k must be an integer between 1 and 20.")
        for name, score, upper in (
            ("min_rrf_score", min_rrf_score, 2 / 61),
            ("min_reranker_score", min_reranker_score, 1.0),
            ("min_reader_score", min_reader_score, 1.0),
        ):
            if not np.isfinite(score) or not 0 <= score <= upper:
                raise ValueError(f"{name} must be between 0 and {upper:.6f}.")

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
        question = validate_question_tokens(question, self.reranker.tokenizer)
        if candidate_k < 1 or rerank_k < 1:
            raise ValueError("candidate_k and rerank_k must be positive.")
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
        self.validate_answer_options(context_k, min_rrf_score, min_reranker_score, min_reader_score)
        evidence = self.search(question)[:context_k]
        if self.reader is None:
            return {"answer": None, "answer_source": "retrieval_only", "evidence": evidence}

        if not evidence:
            return self._abstain(evidence, ["No evidence was retrieved."])
        candidates: list[AnswerCandidate] = []
        rejected = []
        for item in evidence:
            reason = None
            if item["rrf_score"] < min_rrf_score:
                reason = "RRF score below threshold"
            elif item["reranker_score"] < min_reranker_score:
                reason = "Reranker score below threshold"
            elif not self._explicit_years_match(question, item["context"]):
                reason = "Explicit question year missing from evidence"
            if reason:
                rejected.append(f"Chunk {item['chunk_id']}: {reason}.")
                continue
            answer, score = self._read_span(question, item["context"])
            if not answer.strip():
                reason = "Reader preferred no answer or found no valid span"
            elif not np.isfinite(score) or score < min_reader_score:
                reason = "Reader score below threshold"
            elif not self._answer_in_evidence(answer, item["context"]):
                reason = "Answer text missing from cited evidence"
            else:
                candidates.append(AnswerCandidate(answer, score, item["rank"], item["chunk_id"], item["title"], item["context"]))
            if reason:
                rejected.append(f"Chunk {item['chunk_id']}: {reason}.")
        if not candidates:
            return self._abstain(evidence, rejected)
        best = max(candidates, key=lambda candidate: candidate.score)
        return {
            "answer": best.answer,
            "answer_source": "extractive_reader",
            "reader_score": best.score,
            "answer_in_evidence": True,
            "citation": {"title": best.title, "chunk_id": best.chunk_id, "context": best.context},
            "evidence": evidence,
        }
