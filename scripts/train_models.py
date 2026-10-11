"""Prepare matched negative arms and run bounded retriever/reranker training."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

# Support both ``python -m scripts.train_models`` and direct script invocation.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from korean_rag.training import (
    Passage, TrainingConfig, TrainingTriplet, file_sha256, load_initial_model,
    prepare_matched_negatives, train_bounded,
)


def read_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig")
    if text.lstrip().startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError(f"Expected JSON array or JSONL objects: {path}")
    return records


def write_json(path: Path, content) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(content, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def directory_hashes(path: Path) -> dict[str, str]:
    if path.is_file():
        return {path.name: file_sha256(path)}
    return {str(file.relative_to(path)).replace("\\", "/"): file_sha256(file)
            for file in sorted(path.rglob("*")) if file.is_file()}


def prepare(args) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_dir}")
    questions = read_records(args.questions)
    passages = [Passage.from_row(row) for row in read_records(args.corpus)]
    rankings = json.loads(args.rankings.read_text(encoding="utf-8-sig"))
    blocked = (json.loads(args.blocked_documents.read_text(encoding="utf-8-sig"))
               if args.blocked_documents else [])
    arms, report = prepare_matched_negatives(
        questions, passages, rankings, seed=args.seed,
        exclude_same_title=not args.allow_same_title, blocked_document_ids=blocked,
    )
    if not arms["hard"]:
        raise ValueError("No eligible training questions remain after negative filtering")
    args.output_dir.mkdir(parents=True)
    for mode, rows in arms.items():
        write_json(args.output_dir / f"{mode}-triplets.json", [asdict(row) for row in rows])
    report["inputs"] = {"questions_sha256": file_sha256(args.questions),
                        "corpus_sha256": file_sha256(args.corpus),
                        "rankings_sha256": file_sha256(args.rankings)}
    if args.blocked_documents:
        report["inputs"]["blocked_documents_sha256"] = file_sha256(args.blocked_documents)
    write_json(args.output_dir / "preparation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def train(args) -> None:
    train_rows = [TrainingTriplet.from_row(row) for row in read_records(args.train)]
    dev_rows = [TrainingTriplet.from_row(row) for row in read_records(args.dev)]
    raw_config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    values = dict(raw_config.get(args.task, raw_config))
    values["task"] = args.task
    if args.max_optimizer_steps is not None:
        values["max_optimizer_steps"] = args.max_optimizer_steps
    config = TrainingConfig(**values)
    model, tokenizer = load_initial_model(
        args.task, args.initial_model, args.tokenizer, args.architecture_config,
    )
    corpus = ([Passage.from_row(row) for row in read_records(args.development_corpus)]
              if args.development_corpus else None)
    inputs = {"train_sha256": file_sha256(args.train), "dev_sha256": file_sha256(args.dev),
              "configuration_sha256": file_sha256(args.config),
              "initial_model_files": directory_hashes(args.initial_model),
              "training_source_sha256": file_sha256(Path(__file__).resolve().parents[1] / "korean_rag" / "training.py"),
              "cli_source_sha256": file_sha256(Path(__file__).resolve())}
    if args.tokenizer:
        inputs["tokenizer_files"] = directory_hashes(args.tokenizer)
    if args.architecture_config:
        inputs["architecture_config_files"] = directory_hashes(args.architecture_config)
    if args.development_corpus:
        inputs["development_corpus_sha256"] = file_sha256(args.development_corpus)
    report = train_bounded(model, tokenizer, train_rows, dev_rows, args.output_dir, config,
                           device_name=args.device, development_corpus=corpus)
    write_json(args.output_dir / "input_manifest.json", inputs)
    print(json.dumps({key: report[key] for key in (
        "selected_epoch", "selected_development_score", "stopping_reason", "wall_seconds",
    )}, ensure_ascii=False))


def convert(args) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_dir}")
    model, tokenizer = load_initial_model("retriever", args.initial_model,
                                          args.tokenizer, args.architecture_config)
    args.output_dir.mkdir(parents=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    write_json(args.output_dir / "conversion_manifest.json", {
        "source_checkpoint_sha256": file_sha256(args.initial_model),
        "original_architecture_config_files": directory_hashes(args.architecture_config),
        "tokenizer_files": directory_hashes(args.tokenizer),
        "architecture_source": "Explicit original Hugging Face config, never inferred from tensor shapes",
    })


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("prepare", help="Create matched random and hard negative arms")
    command.add_argument("--questions", type=Path, required=True)
    command.add_argument("--corpus", type=Path, required=True)
    command.add_argument("--rankings", type=Path, required=True)
    command.add_argument("--blocked-documents", type=Path)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--seed", type=int, default=378)
    command.add_argument("--allow-same-title", action="store_true")
    command.set_defaults(run=prepare)
    command = commands.add_parser("convert", help="Convert legacy retriever with its explicit original architecture")
    command.add_argument("--initial-model", type=Path, required=True)
    command.add_argument("--tokenizer", type=Path, required=True)
    command.add_argument("--architecture-config", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.set_defaults(run=convert)
    command = commands.add_parser("train", help="Train one bounded arm with development plateau stopping")
    command.add_argument("--task", choices=("retriever", "reranker"), required=True)
    command.add_argument("--train", type=Path, required=True)
    command.add_argument("--dev", type=Path, required=True)
    command.add_argument("--initial-model", type=Path, required=True)
    command.add_argument("--tokenizer", type=Path)
    command.add_argument("--architecture-config", type=Path)
    command.add_argument("--development-corpus", type=Path)
    command.add_argument("--config", type=Path, default=Path("configs/training.json"))
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    command.add_argument("--max-optimizer-steps", type=int)
    command.set_defaults(run=train)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
