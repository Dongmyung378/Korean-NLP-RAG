"""Create project tuning and final-holdout manifests from local KorQuAD JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from korean_rag.benchmark import (SCHEMA_VERSION, build_splits, canonical_sha256,
                                  corpus_records, read_exposure_files, read_korquad,
                                  sha256_file, write_jsonl)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--validation-source", type=Path, required=True)
    parser.add_argument("--exposure-file", type=Path, action="append", required=True)
    parser.add_argument("--protocol", type=Path, default=Path("configs/validation-protocol.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("The output directory must not exist; benchmark records are immutable.")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    split_config = protocol["splits"]
    train = read_korquad(args.train_source, "train")
    validation = read_korquad(args.validation_source, "validation")
    ids, questions, exposure = read_exposure_files(args.exposure_file)
    splits, audit = build_splits(train, validation, ids, questions,
        seed=split_config["seed"], dev_target=split_config["development_question_target"],
        final_target=split_config["final_question_target"], grouping=split_config["grouping"])
    args.output_dir.mkdir(parents=True)
    for name, rows in splits.items():
        write_jsonl(args.output_dir / f"{name}.jsonl", rows)
    write_jsonl(args.output_dir / "corpus.jsonl", corpus_records(validation))
    audit.update({
        "schema_version": SCHEMA_VERSION,
        "source_files": {
            "train": {"file": args.train_source.name, "sha256": sha256_file(args.train_source)},
            "validation": {"file": args.validation_source.name, "sha256": sha256_file(args.validation_source)},
        },
        "historical_exposure_files": exposure,
        "protocol_sha256": sha256_file(args.protocol),
        "manifest_sha256": {path.name: sha256_file(path) for path in sorted(args.output_dir.glob("*.jsonl"))},
        "corpus_paragraph_count": len(corpus_records(validation)),
    })
    audit["benchmark_sha256"] = canonical_sha256(audit)
    with (args.output_dir / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
