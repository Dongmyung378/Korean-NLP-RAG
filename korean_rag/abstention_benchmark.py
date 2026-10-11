"""Auditable synthetic no-answer construction without any model inference."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import json
import math
from pathlib import Path
import unicodedata

from .benchmark import (canonical_sha256, question_key, read_jsonl,
                        score_predictions, sha256_file, text_id,
                        verify_benchmark_directory, write_jsonl)


SUBTYPES = (
    "absent_entity_location", "absent_entity_identity",
    "fabricated_relation", "fictional_future_event",
)
_NAME_SYLLABLES = "가나다라모보서소아오우자하해솔빛누리온별새푸른봄"


def normalized_marker(value: str) -> str:
    """Ignore case, Unicode width, punctuation and spacing for collision checks."""
    return "".join(char for char in unicodedata.normalize("NFKC", value).casefold() if char.isalnum())


def marker_matches_corpus(marker: str, corpus: list[dict]) -> list[str]:
    needle = normalized_marker(marker)
    if not needle:
        raise ValueError("A fictional entity marker cannot be empty.")
    return [str(row["id"]) for row in corpus if needle in normalized_marker(row["text"]) or needle in normalized_marker(row.get("title", ""))]


def _synthetic_name(split: str, index: int, title: str, protocol: Mapping) -> str:
    digest = bytes.fromhex(text_id(f"{protocol['seed']}:{split}:{index}:{title}"))
    nonce = "".join(_NAME_SYLLABLES[value % len(_NAME_SYLLABLES)] for value in digest[:8])
    return f"{title} {protocol['namespace'][split]}{nonce}기록회"


def _clean_title(value: str) -> str:
    # Titles are used as retrieval confusers, not rewritten source documents.
    value = value.replace("\u00b7", " ").replace("\u318d", " ").replace("\u2022", " ")
    return " ".join(value.split())[:48] or "원문 주제"


def _base_cycle(rows: list[dict], seed: int) -> list[dict]:
    """Choose one base QA per paragraph without consulting answers or outcomes."""
    by_document = {}
    for row in sorted(rows, key=lambda item: str(item["query_id"])):
        by_document.setdefault(row["context_id"], row)
    return sorted(by_document.values(), key=lambda row: text_id(f"{seed}:" + row["context_id"]))


def construct_no_answer_sets(corpus: list[dict], base_splits: dict[str, list[dict]], protocol: Mapping) -> tuple[dict[str, list[dict]], dict]:
    """Construct labels by definition and audit the entire declared raw corpus.

    The new name denotes a fresh fictional referent with no aliases. No facts
    about that referent are added to the corpus. Questions explicitly request
    its properties or fabricated relations. The absence check is a collision
    check supporting this construction, not a generic semantic labeler.
    """
    if not corpus:
        raise ValueError("No-answer construction requires a nonempty fixed corpus.")
    dev_documents = {row["context_id"] for row in base_splits["dev"]}
    final_documents = {row["context_id"] for row in base_splits["final"]}
    if dev_documents & final_documents:
        raise ValueError("No-answer base paragraph groups overlap between dev and final.")
    if set(protocol["templates"]["dev"].values()) & set(protocol["templates"]["final"].values()):
        raise ValueError("Development and final surface templates must be disjoint.")
    corpus_ids = {str(row["id"]) for row in corpus}
    sets, seen_entities = {}, set()
    for split, count_key in (("dev", "development_question_count"), ("final", "final_question_count")):
        count = protocol[count_key]
        if not isinstance(count, int) or isinstance(count, bool) or count < 1 or count % len(SUBTYPES):
            raise ValueError("Synthetic question counts must be positive multiples of four.")
        bases = _base_cycle(base_splits[split], protocol["seed"])
        if not bases or any(row["context_id"] not in corpus_ids for row in bases):
            raise ValueError("Every no-answer base paragraph must be present in the fixed corpus.")
        rows = []
        for index in range(count):
            base = bases[index % len(bases)]
            title = _clean_title(base["title"])
            subtype = SUBTYPES[index % len(SUBTYPES)]
            entity = _synthetic_name(split, index, title, protocol)
            event = protocol["event_namespace"][split] + "".join(_NAME_SYLLABLES[value % len(_NAME_SYLLABLES)] for value in bytes.fromhex(text_id(entity))[:5])
            for marker in (entity, event if subtype == "fictional_future_event" else None):
                if marker and marker_matches_corpus(marker, corpus):
                    raise ValueError("A constructed marker accidentally appears in the fixed corpus; abort rather than silently change labels.")
            if normalized_marker(entity) in seen_entities:
                raise ValueError("Constructed entity names must be unique across splits.")
            seen_entities.add(normalized_marker(entity))
            query_id = f"synthetic-no-answer-{split}-{index:03d}"
            question = protocol["templates"][split][subtype].format(title=title, entity=entity, event=event)
            rows.append({
                "query_id": query_id, "id": query_id, "question": question,
                "answerable": False, "answers": [], "gold_answers": [],
                "benchmark_split": split, "category": subtype,
                "base_question_id": base["query_id"], "base_document_id": base["context_id"],
                "base_article_id": base["article_id"], "base_title": base["title"],
                "fictional_entity": entity,
                "fictional_event": event if subtype == "fictional_future_event" else None,
                "surface_template": protocol["templates"][split][subtype],
                "label_provenance": {
                    "type": "synthetic_closed_corpus_construction",
                    "fresh_fictional_referent": True, "declared_aliases": [],
                    "referent_facts_inserted_into_corpus": False,
                    "required_subject_marker": entity,
                    "corpus_paragraphs_checked": len(corpus),
                    "normalized_entity_mentions": 0,
                    "normalized_event_mentions": 0 if subtype == "fictional_future_event" else None,
                    "reason": "The question requests a property or relation of a newly constructed fictional referent. No source fact about that referent or its aliases was supplied. The whole-corpus scan verifies the name did not accidentally collide with an existing mention.",
                    "label_scope": "Unsupported by the declared closed corpus; not a real-world truth judgment or a natural no-answer annotation.",
                },
            })
        sets[split] = rows
    if {question_key(row["question"]) for row in sets["dev"]} & {question_key(row["question"]) for row in sets["final"]}:
        raise ValueError("Synthetic questions overlap across development and final.")
    audit = {
        "counts": {split: len(rows) for split, rows in sets.items()},
        "subtype_counts": {split: dict(Counter(row["category"] for row in rows)) for split, rows in sets.items()},
        "base_paragraph_counts": {split: len({row["base_document_id"] for row in rows}) for split, rows in sets.items()},
        "base_article_overlap": len({row["base_article_id"] for row in sets["dev"]} & {row["base_article_id"] for row in sets["final"]}),
        "base_paragraph_overlap": 0, "entity_name_overlap": 0, "surface_template_overlap": 0,
        "corpus_paragraphs_scanned_per_marker": len(corpus),
        "label_provenance_counts": {"synthetic_constructed": sum(len(rows) for rows in sets.values()), "independent_human_annotation": 0},
        "limitations": list(protocol["limitations"]),
    }
    return sets, audit


def historical_regression_probes(path: Path) -> list[dict]:
    probes = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(probes, list):
        raise ValueError("The historical regression file must contain a list.")
    return [{
        "query_id": f"historical-no-answer-{probe['id']}", "id": f"historical-no-answer-{probe['id']}",
        "question": probe["question"], "answerable": False, "answers": [], "gold_answers": [],
        "benchmark_split": "regression", "category": probe.get("category", "historical_debugging"),
        "original_probe_id": probe["id"],
        "label_provenance": {"type": "historical_manually_selected_debugging_probe", "previously_exposed": True,
                             "label_scope": "Regression debugging only; not part of the held-out synthetic denominator."},
    } for probe in probes]


def prepare_abstention_benchmark(*, benchmark_dir: Path, protocol_path: Path, regression_path: Path, output_dir: Path, review_path: Path | None = None) -> dict:
    benchmark_dir, output_dir = Path(benchmark_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("The no-answer output directory must not exist.")
    benchmark = verify_benchmark_directory(benchmark_dir)
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    corpus = read_jsonl(benchmark_dir / "corpus.jsonl")
    bases = {split: read_jsonl(benchmark_dir / f"{split}.jsonl") for split in ("dev", "final")}
    sets, audit = construct_no_answer_sets(corpus, bases, protocol)
    output_dir.mkdir(parents=True)
    for split, rows in sets.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)
    write_jsonl(output_dir / "regression.jsonl", historical_regression_probes(regression_path))
    if review_path is not None:
        review = json.loads(Path(review_path).read_text(encoding="utf-8"))
        if review.get("corpus_sha256") != benchmark["manifest_sha256"]["corpus.jsonl"]:
            raise ValueError("The regression source review refers to a different corpus.")
        with (output_dir / "regression-source-review.json").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    manifest = {
        "schema_version": 1, **audit,
        "answerable_benchmark_sha256": benchmark["benchmark_sha256"],
        "corpus_sha256": benchmark["manifest_sha256"]["corpus.jsonl"],
        "no_answer_protocol_sha256": sha256_file(protocol_path),
        "historical_probe_source_sha256": sha256_file(regression_path),
        "files_sha256": {path.name: sha256_file(path) for path in sorted(output_dir.iterdir())},
        "final_access_rule": "No final model inference before root freezes the selected model, pipeline configuration and this no-answer manifest. Construction is independent of model outputs.",
    }
    manifest["no_answer_benchmark_sha256"] = canonical_sha256(manifest)
    with (output_dir / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def verify_abstention_directory(directory: Path, protocol_path: Path | None = None) -> dict:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    payload = {key: value for key, value in manifest.items() if key != "no_answer_benchmark_sha256"}
    if canonical_sha256(payload) != manifest.get("no_answer_benchmark_sha256"):
        raise ValueError("The no-answer manifest digest does not match its contents.")
    for name, expected in manifest["files_sha256"].items():
        if Path(name).name != name:
            raise ValueError("No-answer manifest file names cannot contain directory traversal.")
        if not (directory / name).is_file() or sha256_file(directory / name) != expected:
            raise ValueError(f"No-answer data is missing or modified: {name}")
    if protocol_path is not None and sha256_file(protocol_path) != manifest["no_answer_protocol_sha256"]:
        raise ValueError("The no-answer protocol has changed since construction.")
    return manifest


def summarize_no_answer(questions: list[dict], predictions: list[dict]) -> dict:
    if any(row.get("answerable", True) for row in questions):
        raise ValueError("No-answer summaries require exclusively no-answer questions.")
    overall, _ = score_predictions(questions, predictions)
    prediction_lookup = {row["query_id"]: row for row in predictions}
    by_subtype = {}
    for subtype in sorted({row["category"] for row in questions}):
        rows = [row for row in questions if row["category"] == subtype]
        metrics, _ = score_predictions(rows, [prediction_lookup[row["query_id"]] for row in rows])
        by_subtype[subtype] = metrics
    return {**overall, "by_subtype": by_subtype,
            "scope": "Synthetic closed-corpus no-answer stress set; report separately from historical regression and any natural-query annotations."}


def threshold_promotion_decision(candidate: Mapping, baseline: Mapping, validation_protocol: Mapping) -> dict:
    """Enforce the frozen joint quality and coverage rules on development only."""
    fields = ("answerable_exact_match", "answerable_character_f1", "answerable_coverage", "no_answer_false_answer_rate")
    for metrics in (candidate, baseline):
        if any(not isinstance(metrics.get(field), (int, float)) or isinstance(metrics[field], bool) or not math.isfinite(metrics[field]) or not 0 <= metrics[field] <= 1 for field in fields):
            raise ValueError("Threshold promotion requires finite development QA, coverage and no-answer rates.")
    selection = validation_protocol["selection"]
    limits = selection["no_answer_promotion"]
    delta = {field: candidate[field] - baseline[field] for field in fields}
    epsilon = 1e-12
    em_gain = delta["answerable_exact_match"] + epsilon >= selection["minimum_answer_em_gain"]
    f1_gain = delta["answerable_character_f1"] + epsilon >= selection["minimum_answer_f1_gain"] and delta["answerable_exact_match"] + epsilon >= -selection["maximum_em_regression_for_f1_gain"]
    conditions = {
        "declared_answer_gain": em_gain or f1_gain,
        "no_answer_rate_not_worse": delta["no_answer_false_answer_rate"] <= limits["maximum_false_answer_rate_relative_to_baseline"] + epsilon,
        "answerable_coverage_preserved": delta["answerable_coverage"] + epsilon >= limits["minimum_answerable_coverage_relative_to_baseline"],
    }
    return {"promote": all(conditions.values()), "conditions": conditions, "delta": delta,
            "scope": "Development-only promotion. The 10% false-answer aspiration does not override these rules."}
