"""Explicitly convert the recorded original retriever to portable safetensors.

This is an offline preparation command, outside the serving runtime. It never
downloads a checkpoint and only accepts the original hash-pinned project file.
The checkpoint and its original configuration/tokenizer metadata are verified
before the weights-only PyTorch loader is invoked. Existing output is preserved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import tempfile

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from korean_rag.training import file_sha256, load_initial_model


SOURCE_BYTES = 442558247
SOURCE_SHA256 = "ec09ae94fbdf5e8e86e1766353bcb71ea49aa222f4096f4f2ea2bfb65e6c33ca"
OUTPUT_BYTES = 442494816
OUTPUT_SHA256 = "2ce1e42b5a20ca387b3ba81eafb90bdeb7c73907ee6b66ebf9b1c7e17f9d681b"
METADATA_FILES = {
    "config.json": {"bytes": 773, "sha256": "98e76f6d895b6b314678ee70053097c30407f8e748db6d0d9bd18df779b3099c"},
    "tokenizer.json": {"bytes": 752251, "sha256": "5e75543922ab94a075f7f8ca2adec6ddcc4da0b39aa608bfae8be1aca212c7de"},
    "tokenizer_config.json": {"bytes": 670, "sha256": "10fa3530adeec22f3f2678de23a95ea7019fb9875b91da08331c6a51f62c4119"},
}


def reject_links(path: Path) -> None:
    for component in (path, *path.parents):
        if component.is_symlink() or (hasattr(component, "is_junction") and component.is_junction()):
            raise ValueError("Conversion input/output paths must not contain filesystem links")


def verify_file(path: Path, *, size: int, sha256: str) -> None:
    reject_links(path)
    if not path.is_file() or path.stat().st_size != size or file_sha256(path) != sha256:
        raise ValueError(f"Original retriever file size or SHA256 differs from the recorded source: {path.name}")


def convert(*, checkpoint: Path, metadata_directory: Path, output_directory: Path,
            report_path: Path | None = None) -> dict:
    checkpoint = checkpoint.absolute()
    metadata_directory = metadata_directory.absolute()
    output_directory = output_directory.absolute()
    report_path = (report_path or output_directory.with_name(output_directory.name + ".conversion.json")).absolute()
    # This check deliberately precedes importing PyTorch or invoking its loader.
    verify_file(checkpoint, size=SOURCE_BYTES, sha256=SOURCE_SHA256)
    reject_links(metadata_directory)
    if not metadata_directory.is_dir() or {path.name for path in metadata_directory.iterdir()} != set(METADATA_FILES):
        raise ValueError("Original metadata directory must contain exactly config.json, tokenizer.json and tokenizer_config.json")
    for name, properties in METADATA_FILES.items():
        verify_file(metadata_directory / name, size=properties["bytes"], sha256=properties["sha256"])
    for target in (output_directory, report_path):
        reject_links(target)
        if target.exists():
            raise FileExistsError(f"Existing conversion output was preserved: {target}")
    if report_path.is_relative_to(output_directory):
        raise ValueError("The conversion report must remain outside the portable model directory")
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "safetensors")}
    expected_versions = {"torch": "2.14.0", "transformers": "5.17.0", "safetensors": "0.8.0"}
    if any(versions[name].split("+")[0] != expected for name, expected in expected_versions.items()):
        raise ValueError("Exact model serialization requires the pinned project torch, transformers and safetensors versions")
    import torch
    torch.set_num_threads(4)
    model, _ = load_initial_model("retriever", checkpoint, tokenizer_path=metadata_directory,
                                 architecture_config=metadata_directory)
    model.to("cpu")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    reject_links(output_directory.parent)
    with tempfile.TemporaryDirectory(prefix=".original-retriever-", dir=output_directory.parent) as temporary:
        stage = Path(temporary) / "model"
        stage.mkdir()
        model.save_pretrained(stage, safe_serialization=True)
        # Preserve the original recorded metadata bytes instead of regenerating
        # tokenizer/config JSON with potentially platform-dependent formatting.
        for name in METADATA_FILES:
            shutil.copyfile(metadata_directory / name, stage / name)
        if {path.name for path in stage.iterdir()} != {*METADATA_FILES, "model.safetensors"}:
            raise ValueError("Conversion generated an unexpected portable file inventory")
        verify_file(stage / "model.safetensors", size=OUTPUT_BYTES, sha256=OUTPUT_SHA256)
        verify_file(checkpoint, size=SOURCE_BYTES, sha256=SOURCE_SHA256)
        for name, properties in METADATA_FILES.items():
            verify_file(stage / name, size=properties["bytes"], sha256=properties["sha256"])
            verify_file(metadata_directory / name, size=properties["bytes"], sha256=properties["sha256"])
        reject_links(output_directory)
        if output_directory.exists():
            raise FileExistsError("Conversion output appeared concurrently and was preserved")
        stage.rename(output_directory)
    report = {"schema_version": 1, "recorded_at": datetime.now(timezone.utc).isoformat(),
              "method": "Hash-pinned original checkpoint restored on CPU with torch.load(weights_only=True), then serialized as safetensors; no retraining",
              "source": {"bytes": SOURCE_BYTES, "sha256": SOURCE_SHA256},
              "output_model": {"bytes": OUTPUT_BYTES, "sha256": OUTPUT_SHA256},
              "metadata_files": METADATA_FILES, "expected_model_bytes_match": True,
              "python": platform.python_version(), "package_versions": versions,
              "source_script_sha256": file_sha256(Path(__file__).resolve())}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    reject_links(report_path)
    with report_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Manually obtained original week2_dense_retriever.pt")
    parser.add_argument("--metadata", type=Path, required=True, help="Recorded original portable config/tokenizer directory, excluding model weights")
    parser.add_argument("--output", type=Path, required=True, help="New portable Hugging Face model directory")
    parser.add_argument("--report", type=Path, help="New JSON report outside the model directory")
    args = parser.parse_args(argv)
    try:
        result = convert(checkpoint=args.checkpoint, metadata_directory=args.metadata,
                         output_directory=args.output, report_path=args.report)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"Original retriever conversion failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
