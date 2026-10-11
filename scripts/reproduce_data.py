"""Reconstruct the measured benchmark from unchanged official source downloads.

Published recipe files contain IDs, digests and audit metadata, not copied
questions or paragraphs. Derived benchmark records are generated locally and
must match the exact measured JSONL digests before any evaluation is permitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from korean_rag.benchmark import (
    corpus_records, read_korquad, sha256_file, validate_split_separation,
    verify_benchmark_directory, write_jsonl,
)


def fetch_source(source: dict, directory: Path, *, offline: bool = False) -> Path:
    name = source["file"]
    if Path(name).name != name or not name.endswith(".json"):
        raise ValueError("Official source filename must be a simple JSON basename")
    if not source["url"].startswith("https://korquad.github.io/dataset/"):
        raise ValueError("Source downloads must use the official HTTPS KorQuAD dataset URL")
    expected = source["sha256"]
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("Official source digest must be SHA256")
    target = directory / name
    if target.exists():
        if not target.is_file() or sha256_file(target) != expected:
            raise ValueError(f"Existing official source is modified; refusing to overwrite: {target}")
        return target
    if offline:
        raise FileNotFoundError(f"Official source is unavailable offline: {target}")
    directory.mkdir(parents=True, exist_ok=True)
    request = Request(source["url"], headers={"User-Agent": "Korean-NLP-RAG-benchmark-reproduction/1"})
    with urlopen(request, timeout=60) as response:
        content = response.read(100 * 1024 * 1024 + 1)
    if len(content) != source["bytes"] or hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f"Official download size or digest differs from the pinned source: {name}")
    # Exclusive creation also rejects a concurrent writer instead of replacing it.
    with target.open("xb") as stream:
        stream.write(content)
    return target


def reconstruct_splits(train: list[dict], validation: list[dict], recipe: dict) -> dict[str, list[dict]]:
    """Apply published ordered IDs without reselecting or inspecting predictions."""
    source_train = {row["query_id"]: row for row in train}
    source_validation = {row["query_id"]: row for row in validation}
    removed = recipe["train_removed_ids"]
    if len(set(removed)) != len(removed) or not set(removed) <= set(source_train):
        raise ValueError("Removed training IDs must be unique and present in the official train source")
    removed_ids = set(removed)
    splits = {"train": [dict(row) for row in train if row["query_id"] not in removed_ids]}
    selected = set()
    for name in ("dev", "final", "unused"):
        identifiers = recipe["ordered_ids"][name]
        if len(set(identifiers)) != len(identifiers) or selected & set(identifiers):
            raise ValueError("Validation split IDs must be unique and disjoint")
        if not set(identifiers) <= set(source_validation):
            raise ValueError(f"Unknown official validation ID in {name}")
        splits[name] = [dict(source_validation[identifier]) for identifier in identifiers]
        selected.update(identifiers)
    if selected != set(source_validation):
        raise ValueError("Ordered validation split IDs must account for the complete official source")
    for name, rows in splits.items():
        for row in rows:
            row["benchmark_split"] = name
    validate_split_separation(splits, grouping=recipe["grouping"])
    return splits


def reproduce(*, recipe_path: Path, source_dir: Path, output_dir: Path,
              offline: bool = False) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Benchmark output is immutable and must not already exist: {output_dir}")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8-sig"))
    if recipe.get("schema_version") != 1:
        raise ValueError("Unsupported benchmark reproduction recipe schema")
    manifest_name = recipe["benchmark_manifest_file"]
    protocol_name = recipe["protocol_file"]
    if Path(manifest_name).name != manifest_name or Path(protocol_name).name != protocol_name:
        raise ValueError("Recipe config references must be simple basenames")
    manifest_path = recipe_path.parent / manifest_name
    protocol_path = recipe_path.parent / protocol_name
    if sha256_file(manifest_path) != recipe["benchmark_manifest_sha256"]:
        raise ValueError("Published measured benchmark manifest differs from the recipe")
    expected_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(protocol_path) != expected_manifest["protocol_sha256"]:
        raise ValueError("Published protocol differs from the measured benchmark")
    train_path = fetch_source(recipe["sources"]["train"], source_dir, offline=offline)
    validation_path = fetch_source(recipe["sources"]["validation"], source_dir, offline=offline)
    for name, path in (("train", train_path), ("validation", validation_path)):
        if sha256_file(path) != expected_manifest["source_files"][name]["sha256"]:
            raise ValueError("Recipe source digest differs from measured benchmark provenance")
    train = read_korquad(train_path, "train")
    validation = read_korquad(validation_path, "validation")
    splits = reconstruct_splits(train, validation, recipe)
    if {name: len(rows) for name, rows in splits.items()} != expected_manifest["actual_counts"]:
        raise ValueError("Reconstructed split counts differ from the measured benchmark")
    output_dir.mkdir(parents=True)
    for name, rows in splits.items():
        write_jsonl(output_dir / f"{name}.jsonl", rows)
    write_jsonl(output_dir / "corpus.jsonl", corpus_records(validation))
    for name, expected_hash in expected_manifest["manifest_sha256"].items():
        if sha256_file(output_dir / name) != expected_hash:
            raise ValueError(f"Reconstructed file differs from the measured bytes: {name}")
    with (output_dir / "manifest.json").open("xb") as stream:
        stream.write(manifest_path.read_bytes())
    verified = verify_benchmark_directory(output_dir, protocol_path)
    report = {"benchmark_sha256": verified["benchmark_sha256"],
              "recipe_sha256": sha256_file(recipe_path), "source_script_sha256": sha256_file(Path(__file__).resolve()),
              "source_sha256": {"train": sha256_file(train_path), "validation": sha256_file(validation_path)},
              "matched_file_sha256": verified["manifest_sha256"], "actual_counts": verified["actual_counts"],
              "all_files_match_measured_bytes": True,
              "data_distribution": "Unchanged official source downloads; QA text and derived records are constructed only in the user's local directory"}
    with (output_dir / "reproduction.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=Path("configs/benchmark-splits.json"))
    parser.add_argument("--source-dir", type=Path, default=Path(".local/source-data"))
    parser.add_argument("--output-dir", type=Path, default=Path(".local/benchmark-v1"))
    parser.add_argument("--offline", action="store_true", help="Require existing hash-verified official source files")
    args = parser.parse_args(argv)
    print(json.dumps(reproduce(recipe_path=args.recipe, source_dir=args.source_dir,
                              output_dir=args.output_dir, offline=args.offline), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
