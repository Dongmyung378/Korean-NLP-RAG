"""Auditable question splits, complete-span relevance and selective QA metrics.

This module has no model dependency. Benchmark creation never runs inference.
The final split is a project tuning holdout, not the private KorQuAD test set
and not proof that a public pretrained reader has never encountered its data.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
import csv
import hashlib
import json
from math import log2
from pathlib import Path
import unicodedata

from .qa_metrics import score_answer, wilson_interval


SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def text_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_sha256(value: object) -> str:
    return text_id(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False))


def question_key(value: str) -> str:
    """Conservative exact duplicate key, preserving punctuation and semantics."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL records must be objects: {path}")
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping]) -> None:
    """Create a file once; refuse to overwrite existing benchmark evidence."""
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def verify_benchmark_directory(directory: Path, protocol: Path | None = None) -> dict:
    """Reject missing or modified manifests before development or final inference."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    payload = {key: value for key, value in manifest.items() if key != "benchmark_sha256"}
    if canonical_sha256(payload) != manifest.get("benchmark_sha256"):
        raise ValueError("The benchmark manifest digest does not match its contents.")
    for name, expected_hash in manifest["manifest_sha256"].items():
        if Path(name).name != name:
            raise ValueError("Benchmark file names must not contain directory traversal.")
        path = directory / name
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Benchmark file is missing or modified: {name}")
    if protocol is not None and sha256_file(protocol) != manifest["protocol_sha256"]:
        raise ValueError("The benchmark protocol has changed since split creation.")
    return manifest


def freeze_configuration(*, benchmark_sha256: str, protocol_sha256: str,
                         configuration: Mapping, source_sha256: Mapping[str, str],
                         artifact_sha256: Mapping[str, str]) -> dict:
    """Bind selection to exact inputs; the caller persists this before final access."""
    for label, hashes in (("source", source_sha256), ("artifact", artifact_sha256)):
        if not hashes or any(not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value) for value in hashes.values()):
            raise ValueError(f"{label} hashes must be nonempty SHA256 mappings.")
    for value in (benchmark_sha256, protocol_sha256):
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("Benchmark and protocol digests must be SHA256 values.")
    payload = {
        "schema_version": SCHEMA_VERSION, "benchmark_sha256": benchmark_sha256,
        "protocol_sha256": protocol_sha256, "configuration": dict(configuration),
        "source_sha256": dict(source_sha256), "artifact_sha256": dict(artifact_sha256),
    }
    return {**payload, "freeze_sha256": canonical_sha256(payload)}


def verify_final_freeze(frozen: Mapping, *, benchmark_sha256: str, protocol_sha256: str,
                       configuration: Mapping, source_sha256: Mapping[str, str],
                       artifact_sha256: Mapping[str, str]) -> None:
    """Fail closed when an inference dependency differs from the selected bundle."""
    expected = freeze_configuration(benchmark_sha256=benchmark_sha256,
        protocol_sha256=protocol_sha256, configuration=configuration,
        source_sha256=source_sha256, artifact_sha256=artifact_sha256)
    if dict(frozen) != expected:
        raise ValueError("Final inference requires the unchanged predeclared configuration and frozen source, model and benchmark hashes.")


def _qa_record(qa: dict, title: str, context: str, article_id: str, source_split: str) -> dict:
    query_id = str(qa.get("id", qa.get("query_id", ""))).strip()
    question = qa.get("question")
    if not query_id or not isinstance(question, str) or not question.strip():
        raise ValueError("Every QA requires a nonempty ID and question.")
    answers = qa.get("answers", [])
    if isinstance(answers, dict):
        texts, starts = answers.get("text", []), answers.get("answer_start", [])
        if len(texts) != len(starts):
            raise ValueError(f"Answer texts and offsets differ in length: {query_id}")
        answers = [{"text": text, "answer_start": start} for text, start in zip(texts, starts)]
    if not isinstance(answers, list) or not answers:
        raise ValueError(f"KorQuAD v1 examples require gold answers: {query_id}")
    checked = []
    for answer in answers:
        text, start = answer.get("text"), answer.get("answer_start")
        if not isinstance(text, str) or not text or not isinstance(start, int) or isinstance(start, bool) or start < 0:
            raise ValueError(f"Invalid answer annotation: {query_id}")
        if context[start:start + len(text)] != text:
            raise ValueError(f"Gold answer offset mismatch; no automatic relocation is allowed: {query_id}")
        checked.append({"text": text, "answer_start": start})
    return {
        "query_id": query_id, "id": query_id, "question": question, "title": title,
        "source_split": source_split, "article_id": article_id,
        "context_id": text_id(context), "document_id": text_id(context),
        "context_key": hashlib.sha1(context.encode("utf-8")).hexdigest(),
        "context": context, "answers": checked, "gold_answers": [answer["text"] for answer in checked], "answerable": True,
    }


def read_korquad(path: Path, source_split: str) -> list[dict]:
    """Read official nested JSON or an explicitly supplied flattened HF export."""
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    records = []
    if isinstance(value, dict) and isinstance(value.get("data"), list):
        for article in value["data"]:
            title = str(article.get("title", ""))
            article_id = "title:" + text_id(title)
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                if not isinstance(context, str) or not context:
                    raise ValueError("Contexts must be nonempty strings.")
                for qa in paragraph["qas"]:
                    records.append(_qa_record(qa, title, context, article_id, source_split))
    elif isinstance(value, list):
        for qa in value:
            context, title = qa.get("context"), str(qa.get("title", ""))
            if not isinstance(context, str) or not context:
                raise ValueError("Contexts must be nonempty strings.")
            article_id = str(qa.get("article_id", "title:" + text_id(title)))
            records.append(_qa_record(qa, title, context, article_id, source_split))
    else:
        raise ValueError(f"Expected official KorQuAD JSON or a list of flattened records: {path}")
    ids = [row["query_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate question IDs in {path}")
    if not records:
        raise ValueError(f"No QA records in {path}")
    return records


def read_exposure_files(paths: Iterable[Path]) -> tuple[set[str], set[str], list[dict]]:
    """Load exported historical question IDs and text without interpreting results."""
    ids, questions, provenance = set(), set(), []
    for path in paths:
        path = Path(path)
        if path.suffix.lower() == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
        elif path.suffix.lower() == ".jsonl":
            rows = read_jsonl(path)
        else:
            rows = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(rows, dict) and isinstance(rows.get("queries"), list):
                rows = [{"question": question} for question in rows["queries"]]
        if not isinstance(rows, list):
            raise ValueError(f"Exposure file must contain QA rows or a queries list: {path}")
        observed = 0
        for row in rows:
            query_id = row.get("query_id", row.get("id"))
            question = row.get("question", row.get("query"))
            if query_id:
                ids.add(str(query_id))
            if isinstance(question, str) and question.strip():
                questions.add(question_key(question))
            if query_id or question:
                observed += 1
        if not observed:
            raise ValueError(f"No question IDs or text in exposure file: {path}")
        provenance.append({"file": path.name, "sha256": sha256_file(path), "rows": len(rows)})
    return ids, questions, provenance


def grouped_records(records: list[dict], grouping: str = "paragraph") -> list[list[dict]]:
    """Link documents and normalized duplicate questions into indivisible groups."""
    if grouping not in {"paragraph", "article"}:
        raise ValueError("grouping must be paragraph or article.")
    parents = list(range(len(records)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first, second):
        a, b = find(first), find(second)
        if a != b:
            parents[max(a, b)] = min(a, b)

    links = {}
    for index, row in enumerate(records):
        for key in (("paragraph", row["context_id"]), ("question", question_key(row["question"]))):
            if key in links:
                union(index, links[key])
            else:
                links[key] = index
        if grouping == "article":
            key = ("article", row["article_id"])
            if key in links:
                union(index, links[key])
            else:
                links[key] = index
    groups = defaultdict(list)
    for index, row in enumerate(records):
        groups[find(index)].append(row)
    return [sorted(group, key=lambda row: row["query_id"]) for group in groups.values()]


def _ordered_groups(groups: list[list[dict]], seed: int) -> list[list[dict]]:
    return sorted(groups, key=lambda group: text_id(f"{seed}:" + ":".join(row["query_id"] for row in group)))


def _take_groups(groups: list[list[dict]], target: int) -> tuple[list[dict], list[list[dict]]]:
    if not isinstance(target, int) or isinstance(target, bool) or target < 1:
        raise ValueError("Question targets must be positive integers.")
    selected, rest = [], []
    for group in groups:
        if len(selected) < target:
            selected.extend(group)
        else:
            rest.append(group)
    if len(selected) < target:
        raise ValueError(f"Insufficient eligible questions: requested {target}, found {len(selected)}.")
    return selected, rest


def build_splits(train: list[dict], validation: list[dict], exposed_ids: set[str], exposed_questions: set[str], *, seed: int = 378, dev_target: int = 300, final_target: int = 500, grouping: str = "paragraph") -> tuple[dict[str, list[dict]], dict]:
    """Reserve final groups before development; keep known exposure out of final.

    A complete paragraph is the default document unit. Article grouping is an
    explicit stronger option and fails if exposure leaves too few examples.
    """
    combined_ids = [row["query_id"] for row in train + validation]
    if len(combined_ids) != len(set(combined_ids)):
        raise ValueError("Question IDs overlap between source splits.")
    validation_questions = {question_key(row["question"]) for row in validation}
    validation_contexts = {row["context_id"] for row in validation}
    validation_articles = {row["article_id"] for row in validation} if grouping == "article" else set()
    clean_train = [row for row in train if question_key(row["question"]) not in validation_questions and row["context_id"] not in validation_contexts and row["article_id"] not in validation_articles]
    original_train_questions = {question_key(row["question"]) for row in train}
    original_train_contexts = {row["context_id"] for row in train}
    original_train_articles = {row["article_id"] for row in train} if grouping == "article" else set()
    groups = _ordered_groups(grouped_records(validation, grouping), seed)
    exposed, eligible = [], []
    original_training_overlap_groups = 0
    historically_exposed_groups = 0
    for group in groups:
        original_training_overlap = any(question_key(row["question"]) in original_train_questions or row["context_id"] in original_train_contexts or row["article_id"] in original_train_articles for row in group)
        original_training_overlap_groups += int(original_training_overlap)
        historical_exposure = any(row["query_id"] in exposed_ids or question_key(row["question"]) in exposed_questions for row in group)
        historically_exposed_groups += int(historical_exposure)
        destination = exposed if original_training_overlap or historical_exposure else eligible
        destination.append(group)
    final, remainder = _take_groups(eligible, final_target)
    # Remaining untouched groups and historical exposed groups are both legal
    # development data. No final question or source paragraph is used to tune.
    dev, _ = _take_groups(_ordered_groups(exposed + remainder, seed + 1), dev_target)
    used_ids = {row["query_id"] for row in dev + final}
    unused = [row for row in validation if row["query_id"] not in used_ids]
    splits = {"train": clean_train, "dev": dev, "final": final, "unused": unused}
    for split, rows in splits.items():
        for row in rows:
            row["benchmark_split"] = split
    audit = {
        "grouping": grouping, "seed": seed, "target_counts": {"dev": dev_target, "final": final_target},
        "actual_counts": {name: len(rows) for name, rows in splits.items()},
        "source_validation_groups": len(groups), "historically_exposed_groups": historically_exposed_groups,
        "groups_excluded_from_final": len(exposed),
        "eligible_unexposed_groups": len(eligible),
        "train_removed_for_validation_overlap": len(train) - len(clean_train),
        "validation_groups_overlapping_original_training_source": original_training_overlap_groups,
        "known_exposed_validation_questions": sum(row["query_id"] in exposed_ids or question_key(row["question"]) in exposed_questions for row in validation),
        "final_known_exposure_count": 0,
        "final_article_overlap_with_dev": len({row["article_id"] for row in final} & {row["article_id"] for row in dev}),
        "limitations": [
            "This is a holdout from project tuning and historical exported question inference, not the official private KorQuAD test.",
            "The published KorQuAD-finetuned reader may have used this public validation set for model selection.",
            "Historical notebooks inspected aggregate token lengths and answer preservation across the public validation set.",
            "Paragraph grouping does not imply article independence; article overlap is reported explicitly.",
            "All validation source paragraphs, including final gold documents, remain in the fixed retrieval corpus.",
        ],
    }
    validate_split_separation(splits, grouping)
    return splits, audit


def validate_split_separation(splits: dict[str, list[dict]], grouping: str = "paragraph") -> None:
    for first, second in (("train", "dev"), ("train", "final"), ("dev", "final")):
        for field in ("query_id", "context_id", "article_id"):
            if field == "article_id" and grouping != "article":
                continue
            if {row[field] for row in splits[first]} & {row[field] for row in splits[second]}:
                raise ValueError(f"{field} leakage between {first} and {second}.")
        if {question_key(row["question"]) for row in splits[first]} & {question_key(row["question"]) for row in splits[second]}:
            raise ValueError(f"Normalized duplicate questions between {first} and {second}.")


def corpus_records(records: list[dict]) -> list[dict]:
    """Keep the complete raw source paragraphs independently of question splits."""
    corpus = {}
    for row in records:
        item = {key: row[key] for key in ("context_id", "context_key", "article_id", "title", "context", "source_split")}
        item.update({"id": row["context_id"], "document_id": row["context_id"], "text": row["context"]})
        corpus.setdefault(row["context_id"], item)
    return [corpus[key] for key in sorted(corpus)]


def relevant_chunk_ids(question: Mapping, chunks: Iterable[Mapping]) -> list[int]:
    """A gold chunk must contain an entire annotated span in the same paragraph.

    Overlap with only part of an answer is not relevant. An empty result is
    retained and counted as a chunk-boundary coverage failure.
    """
    relevant = []
    for chunk in chunks:
        same_context = chunk.get("context_id") == question.get("context_id") or (
            bool(question.get("context_key")) and chunk.get("context_key") == question["context_key"]
        )
        if not same_context:
            continue
        start, end = int(chunk["char_start"]), int(chunk["char_end"])
        if start < 0 or end < start:
            raise ValueError("Chunk character offsets must be ordered and nonnegative.")
        if any(start <= answer["answer_start"] and end >= answer["answer_start"] + len(answer["text"]) for answer in question["answers"]):
            relevant.append(int(chunk["chunk_id"]))
    return sorted(set(relevant))


def retrieval_metrics(retrieved: list[list[int]], gold: list[list[int]], cutoffs: tuple[int, ...] = (1, 5, 10, 20)) -> dict:
    if len(retrieved) != len(gold) or not retrieved:
        raise ValueError("Retrieval and gold rows must be nonempty and aligned.")
    if any(not isinstance(k, int) or isinstance(k, bool) or k < 1 for k in cutoffs):
        raise ValueError("Retrieval cutoffs must be positive integers.")
    summary = {"question_count": len(gold), "complete_gold_chunk_coverage": sum(bool(ids) for ids in gold) / len(gold)}
    for k in cutoffs:
        hits, reciprocal, ndcg = [], [], []
        for candidates, relevant in zip(retrieved, gold):
            if len(candidates) != len(set(candidates)):
                raise ValueError("Ranked chunk IDs must be unique.")
            matches = [rank for rank, chunk_id in enumerate(candidates[:k], 1) if chunk_id in set(relevant)]
            hits.append(bool(matches))
            reciprocal.append(1 / matches[0] if matches else 0.0)
            actual = sum(1 / log2(rank + 1) for rank in matches)
            ideal = sum(1 / log2(rank + 1) for rank in range(1, min(k, len(set(relevant))) + 1))
            ndcg.append(actual / ideal if ideal else 0.0)
        summary[f"recall_at_{k}"] = sum(hits) / len(gold)
        summary[f"mrr_at_{k}"] = sum(reciprocal) / len(gold)
        summary[f"ndcg_at_{k}"] = sum(ndcg) / len(gold)
    return summary


def score_predictions(questions: list[dict], predictions: list[dict]) -> tuple[dict, list[dict]]:
    """Measure overall quality, selective quality and no-answer safety separately."""
    if not questions:
        raise ValueError("An evaluation set cannot be empty.")
    lookup = {str(row["query_id"]): row for row in predictions}
    expected = {str(row["query_id"]) for row in questions}
    if len(lookup) != len(predictions) or len(expected) != len(questions) or set(lookup) != expected:
        raise ValueError("Predictions must match benchmark IDs exactly, without duplicates.")
    rows = []
    for question in questions:
        prediction = lookup[str(question["query_id"])]
        if "answered" in prediction and not isinstance(prediction["answered"], bool):
            raise ValueError("answered must be a boolean.")
        answered = prediction.get("answered", prediction.get("answer_source") == "extractive_reader")
        answer = prediction.get("answer", "") if answered else ""
        if not isinstance(answer, str):
            raise ValueError("Prediction answers must be strings.")
        if answered and not answer.strip():
            raise ValueError("An accepted answer cannot be empty.")
        answerable = question.get("answerable", True)
        if not isinstance(answerable, bool):
            raise ValueError("answerable must be boolean.")
        if answerable:
            score = score_answer(answer, [annotation["text"] for annotation in question["answers"]])
            if not answered:
                score = {"exact_match": 0.0, "character_f1": 0.0}
        else:
            if question.get("answers"):
                raise ValueError("No-answer examples cannot have positive gold annotations.")
            score = {"exact_match": float(not answered), "character_f1": None}
        rows.append({"query_id": question["query_id"], "answerable": answerable, "answered": answered, "answer": answer, **score})
    answerable_rows = [row for row in rows if row["answerable"]]
    answered_rows = [row for row in answerable_rows if row["answered"]]
    no_answer_rows = [row for row in rows if not row["answerable"]]
    correct = sum(row["exact_match"] for row in answerable_rows)
    no_answer_abstained = sum(not row["answered"] for row in no_answer_rows)
    total, n_answerable = len(rows), len(answerable_rows)
    total_answered = sum(row["answered"] for row in rows)

    def mean(values):
        values = list(values)
        return sum(values) / len(values) if values else None

    summary = {
        "question_count": total, "answerable_count": n_answerable, "no_answer_count": len(no_answer_rows),
        "answered_count": total_answered,
        "answered_answerable_count": len(answered_rows),
        "abstained_answerable_count": n_answerable - len(answered_rows),
        "correct_answerable_count": int(correct),
        "no_answer_abstained_count": no_answer_abstained,
        "no_answer_false_answer_count": len(no_answer_rows) - no_answer_abstained,
        "coverage": mean(row["answered"] for row in rows),
        "answerable_coverage": mean(row["answered"] for row in answerable_rows),
        "answerable_exact_match": mean(row["exact_match"] for row in answerable_rows),
        "answerable_character_f1": mean(row["character_f1"] for row in answerable_rows),
        "answered_answerable_exact_match": mean(row["exact_match"] for row in answered_rows),
        "answered_answerable_character_f1": mean(row["character_f1"] for row in answered_rows),
        "accepted_answer_precision_exact_match": correct / total_answered if total_answered else None,
        "selective_risk_exact_match": 1 - correct / total_answered if total_answered else None,
        "no_answer_abstention_rate": no_answer_abstained / len(no_answer_rows) if no_answer_rows else None,
        "no_answer_false_answer_rate": 1 - no_answer_abstained / len(no_answer_rows) if no_answer_rows else None,
        "no_answer_abstention_wilson_95": wilson_interval(no_answer_abstained, len(no_answer_rows)),
        "balanced_correct_decision_rate": (correct / n_answerable + no_answer_abstained / len(no_answer_rows)) / 2 if n_answerable and no_answer_rows else None,
        "metric_scope": "Gold-normalized EM and Korean character F1; not semantic correctness or an official KorQuAD leaderboard score.",
        "interval_scope": "Wilson interval is descriptive and does not account for related questions or synthetic construction.",
    }
    return summary, rows
