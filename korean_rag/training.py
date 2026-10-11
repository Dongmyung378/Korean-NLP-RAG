"""Bounded, reproducible query-passage training without notebook state.

Data preparation uses the same eligible questions for random and hard negative
arms. Model loading and tensor operations are intentionally lazy so preparation
and split checks can run without PyTorch or Transformers installed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time
import unicodedata
from typing import Any, Mapping, Sequence


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_surface(text: str) -> str:
    """Conservatively match answer surfaces despite spacing or punctuation."""
    return re.sub(r"[^\w]", "", unicodedata.normalize("NFKC", text).casefold())


def answer_texts(row: Mapping[str, Any]) -> tuple[str, ...]:
    values = row.get("gold_answers", row.get("answers", row.get("answer_text", [])))
    if isinstance(values, Mapping):
        values = values.get("text", [])
    if isinstance(values, str):
        values = [values]
    texts = [value.get("text", "") if isinstance(value, Mapping) else value for value in values]
    return tuple(dict.fromkeys(str(value) for value in texts if str(value).strip()))


@dataclass(frozen=True)
class Passage:
    id: str
    document_id: str
    text: str
    title: str = ""

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Passage":
        identifier = str(row.get("id", row.get("chunk_id", "")))
        document_id = str(row.get("document_id", row.get("doc_id", identifier)))
        text = str(row.get("text", row.get("context", "")))
        if not identifier or not document_id or not text.strip():
            raise ValueError("Each passage needs an id, document_id and nonempty text")
        return cls(identifier, document_id, text, str(row.get("title", "")))


@dataclass(frozen=True)
class TrainingTriplet:
    question_id: str
    document_id: str
    question: str
    positive_id: str
    positive: str
    negative_id: str
    negative: str
    answers: tuple[str, ...]
    negative_type: str
    negative_document_id: str = ""

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "TrainingTriplet":
        values = dict(row)
        values["answers"] = tuple(values.get("answers", []))
        result = cls(**values)
        if result.negative_type not in {"random", "hard"}:
            raise ValueError("negative_type must be random or hard")
        if not all((result.question_id, result.document_id, result.question.strip(),
                    result.positive_id, result.positive.strip(), result.negative_id,
                    result.negative.strip(), result.answers)):
            raise ValueError("A training triplet has missing required content")
        if result.positive_id == result.negative_id:
            raise ValueError("Positive and negative passage IDs must differ")
        normalized_answers = tuple(filter(None, map(normalized_surface, result.answers)))
        if not normalized_answers:
            raise ValueError("Annotated answers must contain letters or digits")
        if not any(answer in normalized_surface(result.positive) for answer in normalized_answers):
            raise ValueError("Positive passage does not contain an annotated answer")
        if any(answer in normalized_surface(result.negative) for answer in normalized_answers):
            raise ValueError("Negative passage contains an annotated answer")
        return result


def validate_split_isolation(
    train: Sequence[TrainingTriplet], dev: Sequence[TrainingTriplet],
) -> None:
    """Reject repeated IDs, equivalent questions and shared positive documents."""
    for name, rows in (("train", train), ("dev", dev)):
        identifiers = [row.question_id for row in rows]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"Duplicate question IDs in {name}")
    checks = {
        "question IDs": ({row.question_id for row in train}, {row.question_id for row in dev}),
        "normalized questions": ({normalized_surface(row.question) for row in train},
                                 {normalized_surface(row.question) for row in dev}),
        "positive documents": ({row.document_id for row in train},
                               {row.document_id for row in dev}),
        "training negatives and development positive documents": (
            {row.negative_document_id for row in train if row.negative_document_id},
            {row.document_id for row in dev}),
    }
    for label, (left, right) in checks.items():
        if left & right:
            raise ValueError(f"Train/dev overlap in {label}")


def prepare_matched_negatives(
    questions: Sequence[Mapping[str, Any]], passages: Sequence[Passage],
    rankings: Mapping[str, Sequence[str]], *, seed: int = 378,
    exclude_same_title: bool = True, blocked_document_ids: Sequence[str] = (),
) -> tuple[dict[str, list[TrainingTriplet]], dict[str, Any]]:
    """Create equally sized arms, excluding possible answer-containing negatives.

    Input questions may specify positive_ids (chunk IDs). Otherwise positives are
    selected from their document_id, requiring an annotated answer in the text.
    Hard rankings must be produced by a fixed pre-training retriever. Missing
    valid negatives exclude a question from BOTH arms, preventing sample drift.
    """
    lookup = {passage.id: passage for passage in passages}
    if len(lookup) != len(passages):
        raise ValueError("Passage IDs must be unique")
    blocked = set(map(str, blocked_document_ids))
    pool = [p for p in passages if p.document_id not in blocked]
    normalized_passages = {p.id: normalized_surface(p.text) for p in passages}
    document_passages: dict[str, list[Passage]] = {}
    for passage in pool:
        document_passages.setdefault(passage.document_id, []).append(passage)
    rng = random.Random(seed)
    arms: dict[str, list[TrainingTriplet]] = {"random": [], "hard": []}
    dropped: dict[str, int] = {}
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    for row in questions:
        question_id = str(row.get("id", row.get("question_id", row.get("query_id", ""))))
        question = str(row.get("question", ""))
        document_id = str(row.get("document_id", row.get("doc_id", "")))
        answers = answer_texts(row)
        normalized_question = normalized_surface(question)
        if (not question_id or not normalized_question or not document_id or not answers):
            raise ValueError("Questions require id, document_id, question and answers")
        if question_id in seen_ids or normalized_question in seen_questions:
            raise ValueError("Duplicate IDs or equivalent questions in training input")
        seen_ids.add(question_id)
        seen_questions.add(normalized_question)
        if document_id in blocked:
            raise ValueError("A question's gold document is blocked")
        normalized_answers = tuple(filter(None, map(normalized_surface, answers)))
        explicit = row.get("positive_ids")
        if isinstance(explicit, (str, int)):
            explicit = [explicit]
        positive_candidates = ([lookup[str(identifier)] for identifier in explicit]
                               if explicit is not None else
                               document_passages.get(document_id, []))
        positive_candidates = [p for p in positive_candidates
                               if p.document_id == document_id and any(
                                   answer in normalized_passages[p.id]
                                   for answer in normalized_answers)]
        if not positive_candidates:
            dropped["missing_answer_positive"] = dropped.get("missing_answer_positive", 0) + 1
            continue
        # A stable, compact positive avoids always favoring an arbitrary first chunk.
        positive = min(positive_candidates, key=lambda p: (len(p.text), p.id))
        positive_titles = {p.title for p in positive_candidates if p.title}

        def eligible(passage: Passage) -> bool:
            return (passage.document_id != document_id
                    and passage.document_id not in blocked
                    and (not exclude_same_title or not passage.title
                         or passage.title not in positive_titles)
                    and not any(answer in normalized_passages[passage.id]
                                for answer in normalized_answers))

        valid_pool = [p for p in pool if eligible(p)]
        valid_hard = [lookup[str(identifier)] for identifier in rankings.get(question_id, [])
                      if str(identifier) in lookup and eligible(lookup[str(identifier)])]
        if not valid_pool or not valid_hard:
            reason = "missing_hard_negative" if not valid_hard else "missing_random_negative"
            dropped[reason] = dropped.get(reason, 0) + 1
            continue
        choices = {"random": rng.choice(valid_pool), "hard": valid_hard[0]}
        for mode, negative in choices.items():
            arms[mode].append(TrainingTriplet(
                question_id, document_id, question, positive.id, positive.text,
                negative.id, negative.text, answers, mode, negative.document_id,
            ))
    report = {"input_questions": len(questions), "eligible_questions": len(arms["hard"]),
              "dropped": dropped, "seed": seed, "exclude_same_title": exclude_same_title,
              "answer_filter": "NFKC, casefold, remove non-word characters",
              "same_questions_and_positives": all(
                  (a.question_id, a.positive_id, a.positive) ==
                  (b.question_id, b.positive_id, b.positive)
                  for a, b in zip(arms["random"], arms["hard"], strict=True))}
    return arms, report


@dataclass(frozen=True)
class TrainingConfig:
    task: str = "retriever"
    seed: int = 378
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    epochs: int = 2
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    query_max_length: int = 128
    passage_max_length: int = 256
    pair_max_length: int = 384
    evaluation_batch_size: int = 8
    temperature: float = 0.05
    gradient_checkpointing: bool = True
    mixed_precision: bool = True
    patience: int = 1
    min_delta: float = 0.001
    max_optimizer_steps: int | None = None
    warmup_fraction: float = 0.1

    def __post_init__(self) -> None:
        if self.task not in {"retriever", "reranker"}:
            raise ValueError("task must be retriever or reranker")
        for name in ("epochs", "micro_batch_size", "gradient_accumulation_steps",
                     "query_max_length", "passage_max_length", "pair_max_length",
                     "evaluation_batch_size", "patience"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "temperature"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (not math.isfinite(self.weight_decay) or self.weight_decay < 0
                or not math.isfinite(self.min_delta) or self.min_delta < 0
                or not 0 <= self.warmup_fraction < 1):
            raise ValueError("Invalid weight decay, minimum delta or warmup fraction")
        if self.max_optimizer_steps is not None and (type(self.max_optimizer_steps) is not int
                                                     or self.max_optimizer_steps < 1):
            raise ValueError("max_optimizer_steps must be a positive integer")


class PlateauStopper:
    """Stop on development plateau, with the pre-training model as incumbent."""

    def __init__(self, baseline: float, *, patience: int, min_delta: float):
        if not math.isfinite(baseline):
            raise ValueError("Baseline must be finite")
        self.best = baseline
        self.patience = patience
        self.min_delta = min_delta
        self.stale = 0

    def update(self, score: float) -> tuple[bool, bool]:
        if not math.isfinite(score):
            raise ValueError("Development score must be finite")
        improved = score > self.best + self.min_delta
        if improved:
            self.best = score
            self.stale = 0
        else:
            self.stale += 1
        return improved, self.stale >= self.patience


def accumulation_weight(micro_index: int, *, sample_count: int, micro_batch_size: int,
                        accumulation_steps: int) -> float:
    """Weight microbatch means by actual samples, including a partial last group."""
    sample_start = micro_index * micro_batch_size
    group_start = (micro_index // accumulation_steps) * accumulation_steps * micro_batch_size
    group_samples = min(accumulation_steps * micro_batch_size, sample_count - group_start)
    micro_samples = min(micro_batch_size, sample_count - sample_start)
    if group_samples < 1 or micro_samples < 1:
        raise ValueError("Microbatch index lies outside the training samples")
    return micro_samples / group_samples


def load_initial_model(
    task: str, model_path: Path, tokenizer_path: Path | None = None,
    architecture_config: Path | None = None,
):
    """Restore HF artifacts or a legacy retriever using its original HF config."""
    import torch
    from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    if model_path.is_file():
        if task != "retriever" or tokenizer_path is None or architecture_config is None:
            raise ValueError("Legacy retriever requires tokenizer_path and original architecture_config")
        config = AutoConfig.from_pretrained(architecture_config, local_files_only=True)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        state = checkpoint["model_state_dict"]
        model = AutoModel.from_config(config)
        model.load_state_dict({key.removeprefix("roberta."): value for key, value in state.items()}, strict=True)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    else:
        constructor = AutoModel if task == "retriever" else AutoModelForSequenceClassification
        model = constructor.from_pretrained(model_path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path, local_files_only=True)
    if task == "reranker" and model.config.num_labels != 2:
        raise ValueError("Reranker must have two relevance labels")
    return model, tokenizer


def _embedding(model, tokenizer, texts: Sequence[str], device, max_length: int):
    import torch

    inputs = tokenizer(list(texts), padding=True, truncation=True,
                       max_length=max_length, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    hidden = model(**inputs).last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return torch.nn.functional.normalize(pooled.float(), p=2, dim=1)


def _encode_all(model, tokenizer, texts, device, max_length, batch_size):
    import torch

    outputs = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            outputs.append(_embedding(model, tokenizer, texts[start:start + batch_size],
                                      device, max_length).cpu())
    return torch.cat(outputs)


def evaluate_development(model, tokenizer, dev, config, device, corpus=None) -> dict[str, float]:
    """Evaluate only development inputs; no final-test argument is accepted."""
    import torch

    model.eval()
    if config.task == "retriever":
        lookup = {row.positive_id: Passage(row.positive_id, row.document_id, row.positive)
                  for row in dev}
        if corpus is not None:
            lookup = {p.id: p for p in corpus}
        else:
            for row in dev:
                lookup.setdefault(row.negative_id, Passage(row.negative_id, "negative:" + row.negative_id, row.negative))
        passages = list(lookup.values())
        queries = _encode_all(model, tokenizer, [row.question for row in dev], device,
                              config.query_max_length, config.evaluation_batch_size)
        embeddings = _encode_all(model, tokenizer, [p.text for p in passages], device,
                                 config.passage_max_length, config.evaluation_batch_size)
        rankings = torch.argsort(queries @ embeddings.T, dim=1, descending=True).tolist()
        ranks = []
        for row, ranking in zip(dev, rankings, strict=True):
            rank = next((i for i, index in enumerate(ranking, 1)
                         if passages[index].document_id == row.document_id
                         and any(normalized_surface(a) in normalized_surface(passages[index].text)
                                 for a in row.answers)), None)
            ranks.append(rank)
        result = {f"recall_at_{k}": sum(rank is not None and rank <= k for rank in ranks) / len(dev)
                  for k in (1, 5, 20)}
        result["mrr_at_20"] = sum(1 / rank if rank is not None and rank <= 20 else 0 for rank in ranks) / len(dev)
        result["selection_score"] = result["recall_at_5"] + result["mrr_at_20"] * 0.01
        return result
    correct = []
    with torch.inference_mode():
        for start in range(0, len(dev), config.evaluation_batch_size):
            batch = dev[start:start + config.evaluation_batch_size]
            questions = [row.question for row in batch for _ in range(2)]
            contexts = [context for row in batch for context in (row.positive, row.negative)]
            encoded = tokenizer(questions, contexts, padding=True, truncation="only_second",
                                max_length=config.pair_max_length, return_tensors="pt")
            logits = model(**{key: value.to(device) for key, value in encoded.items()}).logits
            margins = (logits[:, 1] - logits[:, 0]).reshape(-1, 2)
            correct.extend((margins[:, 0] > margins[:, 1]).cpu().tolist())
    score = sum(correct) / len(correct)
    return {"pair_ranking_accuracy": score, "selection_score": score}


def train_bounded(
    model, tokenizer, train: Sequence[TrainingTriplet], dev: Sequence[TrainingTriplet],
    output_dir: Path, config: TrainingConfig, *, device_name: str = "auto",
    development_corpus: Sequence[Passage] | None = None,
) -> dict[str, Any]:
    """Train with a fixed budget and keep the best development checkpoint.

    Callers must provide an unused output directory. The initial checkpoint is
    saved as incumbent, so a plateau never silently replaces it with a worse one.
    The downstream pipeline must independently compare candidates on its fixed
    development metric before promotion; pair ranking is only a training metric.
    """
    import torch

    if not train or not dev:
        raise ValueError("Nonempty training and development triplets are required")
    for row in (*train, *dev):
        TrainingTriplet.from_row(asdict(row))
    validate_split_isolation(train, dev)
    if len({row.negative_type for row in train}) != 1:
        raise ValueError("One training run must use exactly one negative arm")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite training output: {output_dir}")
    output_dir.mkdir(parents=True)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available()
                          else "cpu" if device_name == "auto" else device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    use_amp = config.mixed_precision and device.type == "cuda"
    model = model.to(device)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    micro_batches = math.ceil(len(train) / config.micro_batch_size)
    epoch_steps = math.ceil(micro_batches / config.gradient_accumulation_steps)
    total_steps = min(config.epochs * epoch_steps, config.max_optimizer_steps or config.epochs * epoch_steps)
    warmup_steps = int(total_steps * config.warmup_fraction)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:
        (step + 1) / max(1, warmup_steps) if step < warmup_steps else
        max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps)))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    started = time.perf_counter()
    baseline = evaluate_development(model, tokenizer, dev, config, device, development_corpus)
    stopper = PlateauStopper(baseline["selection_score"], patience=config.patience, min_delta=config.min_delta)
    best_dir = output_dir / "best_model"
    model.save_pretrained(best_dir, safe_serialization=True)
    tokenizer.save_pretrained(best_dir)
    history = []
    optimizer_steps = 0
    selected_epoch = 0
    stopping_reason = "epoch_budget"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, config.epochs + 1):
        model.train()
        indices = list(range(len(train)))
        random.Random(config.seed + epoch).shuffle(indices)
        optimizer.zero_grad(set_to_none=True)
        epoch_losses = []
        epoch_started = time.perf_counter()
        for micro_index, start in enumerate(range(0, len(indices), config.micro_batch_size)):
            rows = [train[i] for i in indices[start:start + config.micro_batch_size]]
            loss_weight = accumulation_weight(micro_index, sample_count=len(train),
                                              micro_batch_size=config.micro_batch_size,
                                              accumulation_steps=config.gradient_accumulation_steps)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                if config.task == "retriever":
                    queries = _embedding(model, tokenizer, [row.question for row in rows], device, config.query_max_length)
                    positives = _embedding(model, tokenizer, [row.positive for row in rows], device, config.passage_max_length)
                    negatives = _embedding(model, tokenizer, [row.negative for row in rows], device, config.passage_max_length)
                    logits = torch.stack(((queries * positives).sum(dim=1), (queries * negatives).sum(dim=1)), dim=1)
                    loss = torch.nn.functional.cross_entropy(logits / config.temperature,
                                                             torch.zeros(len(rows), dtype=torch.long, device=device))
                else:
                    questions = [row.question for row in rows for _ in range(2)]
                    contexts = [text for row in rows for text in (row.positive, row.negative)]
                    encoded = tokenizer(questions, contexts, padding=True, truncation="only_second",
                                        max_length=config.pair_max_length, return_tensors="pt")
                    logits = model(**{key: value.to(device) for key, value in encoded.items()}).logits
                    labels = torch.tensor([label for _ in rows for label in (1, 0)], device=device)
                    loss = torch.nn.functional.cross_entropy(logits.float(), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss; run stopped")
            epoch_losses.append(float(loss.detach().cpu()))
            scaler.scale(loss * loss_weight).backward()
            if (micro_index + 1) % config.gradient_accumulation_steps == 0 or micro_index + 1 == micro_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_steps += 1
                if optimizer_steps % 10 == 0:
                    print(json.dumps({"epoch": epoch, "optimizer_steps": optimizer_steps,
                                      "latest_loss": epoch_losses[-1],
                                      "elapsed_seconds": time.perf_counter() - started}), flush=True)
                if optimizer_steps >= total_steps:
                    break
        metrics = evaluate_development(model, tokenizer, dev, config, device, development_corpus)
        improved, plateau = stopper.update(metrics["selection_score"])
        history.append({"epoch": epoch, "optimizer_steps": optimizer_steps,
                        "mean_micro_batch_loss": sum(epoch_losses) / len(epoch_losses),
                        "epoch_seconds": time.perf_counter() - epoch_started,
                        "development": metrics, "improved": improved})
        if improved:
            selected_epoch = epoch
            model.save_pretrained(best_dir, safe_serialization=True)
            tokenizer.save_pretrained(best_dir)
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)
        if plateau:
            stopping_reason = "development_plateau"
            break
        if optimizer_steps >= total_steps:
            stopping_reason = "optimizer_step_budget"
            break
    report = {"configuration": asdict(config), "train_questions": len(train), "dev_questions": len(dev),
              "negative_type": train[0].negative_type, "baseline_development": baseline,
              "history": history, "selected_epoch": selected_epoch,
              "selected_development_score": stopper.best, "stopping_reason": stopping_reason,
              "optimizer_steps": optimizer_steps, "wall_seconds": time.perf_counter() - started,
              "device": str(device), "mixed_precision_used": use_amp,
              "torch_version": torch.__version__,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
              "evaluation_scope": "development only; downstream promotion requires a separate fixed RAG development comparison"}
    (output_dir / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
