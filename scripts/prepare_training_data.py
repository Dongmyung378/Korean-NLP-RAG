"""Build isolated training groups and mine matched negatives reproducibly.

The source is the frozen benchmark TRAIN split, never project development or
final records. Natural sentence boundaries guide bounded token windows. Gold
offsets label complete positive chunks but do not determine chunk boundaries.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from korean_rag.training import (
    Passage, file_sha256, load_initial_model, normalized_surface,
    prepare_matched_negatives, validate_split_isolation,
)
from scripts.train_models import directory_hashes, read_records, write_json


def write_jsonl(path: Path, rows) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def natural_chunks(text, tokenizer, *, max_tokens=256, overlap_tokens=32):
    """Use content-only boundaries and preserve source character offsets."""
    budget = max_tokens - tokenizer.num_special_tokens_to_add(pair=False)
    if budget < 8 or not 0 <= overlap_tokens < budget // 2:
        raise ValueError("Invalid token window or overlap budget")
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
    if not offsets:
        return []
    token_ends = [pair[1] for pair in offsets]
    sentence_ends = sorted({bisect_right(token_ends, match.start())
                            for match in re.finditer(r"(?<=[.!?。！？])\s+|\n+", text)} | {len(offsets)})
    chunks = []
    start = 0
    while start < len(offsets):
        limit = min(start + budget, len(offsets))
        preferred = [position for position in sentence_ends if start + budget // 2 <= position <= limit]
        end = max(preferred) if preferred else limit
        char_start, char_end = offsets[start][0], offsets[end - 1][1]
        window = text[char_start:char_end]
        if len(tokenizer(window, add_special_tokens=True)["input_ids"]) > max_tokens:
            raise ValueError("Re-tokenized chunk exceeds declared maximum")
        chunks.append({"text": window, "start_char": char_start, "end_char": char_end,
                       "token_count": end - start})
        if end == len(offsets):
            break
        start = max(start + 1, end - overlap_tokens)
    return chunks


def choose_groups(rows, *, seed, train_target, dev_target):
    """Deduplicate questions, then select whole paragraphs in stable seed order."""
    seen_ids, seen_questions = set(), set()
    grouped = {}
    dropped_duplicates = 0
    for row in sorted(rows, key=lambda record: str(record["id"])):
        identifier = str(row["id"])
        key = normalized_surface(row["question"])
        if identifier in seen_ids or key in seen_questions:
            dropped_duplicates += 1
            continue
        seen_ids.add(identifier)
        seen_questions.add(key)
        grouped.setdefault(str(row["document_id"]), []).append(row)
    groups = sorted(grouped.values(), key=lambda group:
                    hashlib.sha256(f"{seed}:{group[0]['document_id']}".encode()).hexdigest())
    selected = {"train": [], "dev": []}
    for group in groups:
        if len(selected["dev"]) < dev_target:
            selected["dev"].extend(group)
        elif len(selected["train"]) < train_target:
            selected["train"].extend(group)
        else:
            break
    if len(selected["train"]) < train_target or len(selected["dev"]) < dev_target:
        raise ValueError("Insufficient paragraph-disjoint questions for requested training targets")
    return selected, dropped_duplicates


def build(args):
    from transformers import AutoTokenizer

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_dir}")
    rows = read_records(args.train_source)
    if any(row.get("source_split") not in {"train", "official_train"} for row in rows):
        raise ValueError("Training preparation accepts only official training source records")
    selected, duplicates = choose_groups(rows, seed=args.seed, train_target=args.train_questions,
                                         dev_target=args.dev_questions)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer with character offsets is required")
    output = {}
    dropped = {}
    for split, questions in selected.items():
        documents = {str(row["document_id"]): row for row in questions}
        passages = []
        by_document = {}
        for document_id, row in sorted(documents.items()):
            context = str(row["context"])
            chunks = natural_chunks(context, tokenizer, max_tokens=args.chunk_max_tokens,
                                    overlap_tokens=args.overlap_tokens)
            by_document[document_id] = []
            for index, chunk in enumerate(chunks):
                record = {"id": f"{document_id}:{index:04d}", "document_id": document_id,
                          "title": row.get("title", ""), **chunk}
                passages.append(record)
                by_document[document_id].append(record)
        retained = []
        for row in questions:
            positives = [chunk["id"] for chunk in by_document[str(row["document_id"])]
                         if any(chunk["start_char"] <= answer["answer_start"]
                                and answer["answer_start"] + len(answer["text"]) <= chunk["end_char"]
                                for answer in row["answers"])]
            if positives:
                retained.append({**row, "positive_ids": positives})
        dropped[split] = len(questions) - len(retained)
        output[split] = {"questions": retained, "passages": passages}
    train_docs = {row["document_id"] for row in output["train"]["questions"]}
    dev_docs = {row["document_id"] for row in output["dev"]["questions"]}
    if train_docs & dev_docs:
        raise ValueError("Training and internal development paragraphs overlap")
    args.output_dir.mkdir(parents=True)
    for split, content in output.items():
        write_jsonl(args.output_dir / f"{split}-questions.jsonl", content["questions"])
        write_jsonl(args.output_dir / f"{split}-corpus.jsonl", content["passages"])
    report = {"source_sha256": file_sha256(args.train_source), "source_questions": len(rows),
              "source_scope": "Frozen official-training records only; project development and final never loaded",
              "tokenizer_files": directory_hashes(args.tokenizer), "seed": args.seed,
              "train_target": args.train_questions, "dev_target": args.dev_questions,
              "train_questions": len(output["train"]["questions"]), "dev_questions": len(output["dev"]["questions"]),
              "train_documents": len(train_docs), "dev_documents": len(dev_docs),
              "train_chunks": len(output["train"]["passages"]), "dev_chunks": len(output["dev"]["passages"]),
              "duplicate_questions_removed": duplicates, "questions_without_complete_span": dropped,
              "chunk_max_tokens": args.chunk_max_tokens, "overlap_tokens": args.overlap_tokens,
              "chunking": "Sentence-boundary-aware token windows; boundaries do not use answer offsets",
              "paragraph_overlap": 0,
              "source_script_sha256": file_sha256(Path(__file__).resolve())}
    report["output_sha256"] = directory_hashes(args.output_dir)
    write_json(args.output_dir / "data_manifest.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def mine(args):
    """Use a fixed original encoder, then construct equally sized negative arms."""
    import numpy as np
    import torch
    from korean_rag.retrieval import RobertaDenseRetriever

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_dir}")
    model, tokenizer = load_initial_model("retriever", args.initial_model)
    device = torch.device(args.device)
    encoder = RobertaDenseRetriever(model, tokenizer, device, args.max_length)
    sets = {}
    input_hashes = {}
    for split in ("train", "dev"):
        question_path = args.data_dir / f"{split}-questions.jsonl"
        corpus_path = args.data_dir / f"{split}-corpus.jsonl"
        questions = read_records(question_path)
        passages = [Passage.from_row(row) for row in read_records(corpus_path)]
        embeddings = encoder.encode([p.text for p in passages], batch_size=args.batch_size)
        query_embeddings = encoder.encode([row["question"] for row in questions], batch_size=args.batch_size)
        scores = query_embeddings @ embeddings.T
        top_k = min(args.top_k, len(passages))
        # Stable ties make candidate mining reproducible across repeat executions.
        indices = np.argsort(-scores, axis=1, kind="stable")[:, :top_k]
        rankings = {str(row["id"]): [passages[index].id for index in ranking]
                    for row, ranking in zip(questions, indices, strict=True)}
        arms, preparation_report = prepare_matched_negatives(questions, passages, rankings, seed=args.seed)
        if not arms["hard"]:
            raise ValueError(f"No valid matched negatives for {split}")
        sets[split] = (rankings, arms, preparation_report)
        input_hashes[question_path.name] = file_sha256(question_path)
        input_hashes[corpus_path.name] = file_sha256(corpus_path)
    validate_split_isolation(sets["train"][1]["hard"], sets["dev"][1]["hard"])
    args.output_dir.mkdir(parents=True)
    for split, (rankings, arms, report) in sets.items():
        write_json(args.output_dir / f"{split}-rankings.json", rankings)
        for mode, triplets in arms.items():
            write_json(args.output_dir / f"{split}-{mode}-triplets.json", [asdict(row) for row in triplets])
        write_json(args.output_dir / f"{split}-preparation.json", report)
    report = {"inputs": input_hashes, "initial_model_files": directory_hashes(args.initial_model),
              "mining_model": "Frozen original project STS retriever, masked mean pooling and L2 normalization",
              "max_length": args.max_length, "top_k": args.top_k, "batch_size": args.batch_size,
              "seed": args.seed, "device": str(device), "torch_version": torch.__version__,
              "train_eligible": len(sets["train"][1]["hard"]), "dev_eligible": len(sets["dev"][1]["hard"]),
              "source_script_sha256": file_sha256(Path(__file__).resolve())}
    report["output_sha256"] = directory_hashes(args.output_dir)
    write_json(args.output_dir / "mining_manifest.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("build", help="Prepare grouped training and internal development natural chunks")
    command.add_argument("--train-source", type=Path, required=True)
    command.add_argument("--tokenizer", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--train-questions", type=int, default=3000)
    command.add_argument("--dev-questions", type=int, default=200)
    command.add_argument("--chunk-max-tokens", type=int, default=256)
    command.add_argument("--overlap-tokens", type=int, default=32)
    command.add_argument("--seed", type=int, default=378)
    command.set_defaults(run=build)
    command = commands.add_parser("mine", help="Mine frozen dense rankings and matched negative arms")
    command.add_argument("--data-dir", type=Path, required=True)
    command.add_argument("--initial-model", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    command.add_argument("--max-length", type=int, default=64)
    command.add_argument("--top-k", type=int, default=50)
    command.add_argument("--batch-size", type=int, default=16)
    command.add_argument("--seed", type=int, default=378)
    command.set_defaults(run=mine)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
