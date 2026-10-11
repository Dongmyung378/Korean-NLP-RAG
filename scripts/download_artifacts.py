"""Download and verify a portable model bundle using hash-pinned public files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_UNPACKED_BYTES = 8 * 1024**3
MAX_MEMBERS = 10000
MAX_DIRECT_FILES = 64
FORBIDDEN_SUFFIXES = {".pkl", ".pickle", ".pt", ".pth", ".bin", ".py", ".exe", ".dll", ".bat", ".sh", ".ps1"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def safe_relative(name: str) -> Path:
    """Reject Windows paths, traversal and ambiguous ZIP names on every OS."""
    if not isinstance(name, str) or not name or any(c in name for c in '\\:<>"|?*') or any(ord(c) < 32 for c in name):
        raise ValueError("Invalid relative artifact path")
    components = name.split("/")
    if any(component in {"", ".", ".."} or component.endswith((" ", ".")) for component in components):
        raise ValueError(f"Unsafe relative artifact path: {name}")
    if PurePosixPath(name).is_absolute():
        raise ValueError(f"Absolute artifact path: {name}")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if any(component.split(".")[0].upper() in reserved for component in components):
        raise ValueError(f"Reserved artifact path: {name}")
    relative = Path(*components)
    if relative.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise ValueError(f"Unsafe serialized or executable artifact: {name}")
    return relative


def confined(root: Path, relative: str) -> Path:
    target = root / safe_relative(relative)
    resolved_root = root.resolve()
    resolved = target.resolve()
    if resolved == resolved_root or not resolved.is_relative_to(resolved_root):
        raise ValueError(f"Artifact path escapes its destination: {relative}")
    for ancestor in [target, *target.parents]:
        if ancestor == root.parent:
            break
        if ancestor.is_symlink() or (hasattr(ancestor, "is_junction") and ancestor.is_junction()):
            raise ValueError(f"Artifact path contains a filesystem link: {relative}")
    return target


def read_manifest(root: Path) -> dict[str, dict]:
    path = root / "manifest.json"
    if path.stat().st_size > 5 * 1024**2:
        raise ValueError("Artifact manifest exceeds the size limit")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict) or not files or len(files) > MAX_MEMBERS:
        raise ValueError("manifest.json must contain a non-empty files mapping")
    for name, properties in files.items():
        safe_relative(name)
        if not isinstance(properties, dict) or set(properties).isdisjoint({"bytes", "size"}):
            raise ValueError(f"Missing file size in manifest: {name}")
        hash_value = properties.get("sha256", "")
        if not isinstance(hash_value, str) or len(hash_value) != 64 or any(c not in "0123456789abcdef" for c in hash_value):
            raise ValueError(f"Invalid SHA256 in manifest: {name}")
        size = properties.get("bytes", properties.get("size"))
        if not isinstance(size, int) or isinstance(size, bool) or not 0 <= size <= MAX_UNPACKED_BYTES:
            raise ValueError(f"Invalid file size in manifest: {name}")
    return files


def verify_bundle(root: Path) -> dict:
    if root.is_symlink() or (hasattr(root, "is_junction") and root.is_junction()):
        raise ValueError("Bundle root must not be a filesystem link")
    root = root.resolve()
    if not root.is_dir():
        raise ValueError("Artifact bundle directory does not exist")
    if (root / "manifest.json").is_symlink():
        raise ValueError("Artifact manifest must be a regular file")
    files = read_manifest(root)
    required = {"config.json", "corpus.jsonl", "chunks.jsonl", "dense.index"}
    if not required.issubset(files):
        raise ValueError("Bundle manifest omits required configuration/corpus/index files")
    for kind in ("retriever", "reranker", "reader"):
        prefix = f"models/{kind}/"
        if not any(name.startswith(prefix) and name.endswith(".safetensors") for name in files):
            raise ValueError(f"Bundle has no safetensors model for {kind}")
    total = 0
    expected_names = {"manifest.json", *files}
    actual_names = set()
    for path in root.rglob("*"):
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise ValueError("Bundle contains a filesystem link")
        if path.is_file():
            actual_names.add(path.relative_to(root).as_posix())
    if actual_names != expected_names:
        raise ValueError("Bundle files do not match the declared manifest")
    for name, properties in files.items():
        path = confined(root, name)
        size = properties.get("bytes", properties.get("size"))
        if not path.is_file() or path.stat().st_size != size or digest(path) != properties["sha256"]:
            raise ValueError(f"Artifact checksum or size mismatch: {name}")
        total += size
        if total > MAX_UNPACKED_BYTES:
            raise ValueError("Artifact bundle exceeds the size limit")
    return {"valid": True, "files": len(files), "bytes": total, "manifest_sha256": digest(root / "manifest.json")}


def extract_archive(archive: Path, target: Path, seen: set[str]) -> None:
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        if len(members) + len(seen) > MAX_MEMBERS:
            raise ValueError("Archive contains too many entries")
        current_bytes = sum(path.stat().st_size for path in target.rglob("*") if path.is_file())
        total = current_bytes + sum(info.file_size for info in members)
        if total > MAX_UNPACKED_BYTES:
            raise ValueError("Uncompressed archive exceeds the size limit")
        for info in members:
            name = info.filename[:-1] if info.is_dir() else info.filename
            relative = safe_relative(name)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR}):
                raise ValueError(f"Archive contains a filesystem link or special file: {name}")
            destination = confined(target, name)
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            folded = relative.as_posix().casefold()
            if folded in seen:
                raise ValueError(f"Duplicate artifact path: {name}")
            seen.add(folded)
            destination.parent.mkdir(parents=True, exist_ok=True)
            confined(target, name)
            with source.open(info) as input_stream, destination.open("xb") as output_stream:
                shutil.copyfileobj(input_stream, output_stream, length=1024**2)


def validate_download_entry(entry: dict) -> tuple[str, int, str]:
    if not isinstance(entry, dict):
        raise ValueError("Download entries must be objects")
    url = entry.get("url", "")
    if not isinstance(url, str):
        raise ValueError("Download URLs must be strings")
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Download URLs must be HTTPS without embedded credentials")
    size, sha256 = entry.get("bytes"), entry.get("sha256", "")
    if not isinstance(size, int) or isinstance(size, bool) or not 0 < size <= MAX_UNPACKED_BYTES:
        raise ValueError("Download size must be declared and within the size limit")
    if not isinstance(sha256, str) or len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
        raise ValueError("Download SHA256 must be declared")
    return url, size, sha256


def download_archive(entry: dict, destination: Path) -> None:
    """Stream either an archive or an individual artifact with exact validation."""
    url, size, sha256 = validate_download_entry(entry)
    value = hashlib.sha256()
    count = 0
    request = Request(url, headers={"User-Agent": "Korean-NLP-RAG/2.0 artifact verification"})
    with urlopen(request, timeout=120) as response, destination.open("xb") as stream:
        redirected = urlparse(response.geturl())
        if redirected.scheme != "https" or not redirected.hostname or redirected.username or redirected.password:
            raise ValueError("Download redirect is not credential-free HTTPS")
        for block in iter(lambda: response.read(1024**2), b""):
            count += len(block)
            if count > size:
                raise ValueError("Download exceeds the declared size")
            value.update(block)
            stream.write(block)
    if count != size or value.hexdigest() != sha256:
        raise ValueError("Download checksum or size mismatch")


def materialize_source_data(recipe: dict, bundle: Path, source: Path) -> None:
    """Reconstruct source files locally without redistributing modified KorQuAD."""
    if not isinstance(recipe, dict) or not isinstance(recipe.get("source"), dict):
        raise ValueError("A data recipe requires a hash-pinned official source")
    if (bundle / "corpus.jsonl").exists() or (bundle / "chunks.jsonl").exists():
        raise ValueError("Recipe-based archives must omit corpus.jsonl and chunks.jsonl")
    chunks_recipe = recipe.get("chunks")
    if not isinstance(chunks_recipe, list) or not 1 <= len(chunks_recipe) <= 200000:
        raise ValueError("A data recipe requires a bounded, nonempty chunk list")
    source_split = recipe.get("source_split", "validation")
    if not isinstance(source_split, str) or not source_split:
        raise ValueError("The data recipe requires a source split name")
    chunks_newline = recipe.get("chunks_newline", "\n")
    if chunks_newline not in {"\n", "\r\n"}:
        raise ValueError("The chunk recipe newline must be LF or CRLF")
    download_archive(recipe["source"], source)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from korean_rag.benchmark import read_korquad, corpus_records, write_jsonl
    corpus = corpus_records(read_korquad(source, source_split))
    parents = {row["context_id"]: row for row in corpus}
    write_jsonl(bundle / "corpus.jsonl", corpus)
    with (bundle / "chunks.jsonl").open("x", encoding="utf-8", newline=chunks_newline) as stream:
        for index, recipe_row in enumerate(chunks_recipe):
            if not isinstance(recipe_row, dict) or recipe_row.get("chunk_id") != index or isinstance(recipe_row.get("chunk_id"), bool):
                raise ValueError("Chunk recipe IDs must be consecutive integers")
            context_id = recipe_row.get("context_id")
            if not isinstance(context_id, str) or context_id not in parents:
                raise ValueError("Chunk recipe refers to an absent source paragraph")
            parent = parents[context_id]
            start, end = recipe_row.get("char_start"), recipe_row.get("char_end")
            if not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int) or isinstance(end, bool) or not 0 <= start < end <= len(parent["context"]):
                raise ValueError("Chunk recipe offsets must remain inside the original source")
            row = {"chunk_id": index, "context_id": context_id, "title": parent["title"],
                   "char_start": start, "char_end": end, "context": parent["context"][start:end]}
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def install_local_retriever(bundle: Path, source: Path) -> None:
    """Accept only the exact converted original weight named by the manifest."""
    name = "models/retriever/model.safetensors"
    files = read_manifest(bundle)
    if name not in files:
        raise ValueError("Local retriever requires its exact safetensors path in the bundle manifest")
    destination = confined(bundle, name)
    if destination.exists():
        raise ValueError("Local retriever collides with an archive or downloaded model file")
    source = source.absolute()
    for component in (source, *source.parents):
        if component.is_symlink() or (hasattr(component, "is_junction") and component.is_junction()):
            raise ValueError("Local retriever source must not contain filesystem links")
    expected = files[name]
    expected_size = expected.get("bytes", expected.get("size"))
    if not source.is_file() or not stat.S_ISREG(source.stat().st_mode) or source.suffix.lower() != ".safetensors":
        raise ValueError("Local retriever must be a regular safetensors file")
    if source.stat().st_size != expected_size or digest(source) != expected["sha256"]:
        raise ValueError("Local retriever SHA256 or size differs from the recorded manifest")
    current_bytes = sum(path.stat().st_size for path in bundle.rglob("*") if path.is_file())
    if current_bytes + expected_size > MAX_UNPACKED_BYTES:
        raise ValueError("Local retriever and existing artifacts exceed the total size limit")
    destination.parent.mkdir(parents=True, exist_ok=True)
    confined(bundle, name)
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream, length=1024**2)
    if destination.stat().st_size != expected_size or digest(destination) != expected["sha256"]:
        raise ValueError("Local retriever changed during copying")


def install_bundle(download_manifest: Path, destination: Path, *, local_retriever: Path | None = None) -> dict:
    destination = destination.absolute()
    if destination.exists():
        if local_retriever is not None:
            raise ValueError("Local retriever preparation requires a new bundle destination; the existing bundle was preserved")
        return {"reused": True, **verify_bundle(destination)}
    descriptor = json.loads(download_manifest.read_text(encoding="utf-8"))
    archives = descriptor.get("archives")
    if descriptor.get("schema_version") != 1 or not isinstance(archives, list) or not 1 <= len(archives) <= 8:
        raise ValueError("Download manifest requires schema_version 1 and one to eight archives")
    direct_files = descriptor.get("files", [])
    if not isinstance(direct_files, list) or len(direct_files) > MAX_DIRECT_FILES:
        raise ValueError(f"Individual artifact downloads require a list of at most {MAX_DIRECT_FILES} files")
    direct_paths: set[str] = set()
    declared_total = 0
    for entry in archives:
        _, size, _ = validate_download_entry(entry)
        declared_total += size
    for entry in direct_files:
        _, size, _ = validate_download_entry(entry)
        name = entry.get("path")
        relative = safe_relative(name)
        folded = relative.as_posix().casefold()
        if folded in direct_paths:
            raise ValueError(f"Duplicate individual artifact path: {name}")
        direct_paths.add(folded)
        declared_total += size
    if declared_total > MAX_UNPACKED_BYTES:
        raise ValueError("Declared downloads exceed the total size limit")
    if destination.parent.resolve() != destination.parent:
        raise ValueError("Destination parent must not contain filesystem links")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.resolve() != destination.parent:
        raise ValueError("Destination parent must not contain filesystem links")
    with tempfile.TemporaryDirectory(prefix=".rag-download-", dir=destination.parent) as temp:
        stage = Path(temp)
        bundle = stage / "bundle"
        bundle.mkdir()
        seen: set[str] = set()
        for index, entry in enumerate(archives):
            archive = stage / f"archive-{index}.zip"
            download_archive(entry, archive)
            extract_archive(archive, bundle, seen)
        if seen & direct_paths:
            raise ValueError("Individual artifact download collides with an archive file")
        for entry in direct_files:
            destination_file = confined(bundle, entry["path"])
            current_bytes = sum(path.stat().st_size for path in bundle.rglob("*") if path.is_file())
            if current_bytes + entry["bytes"] > MAX_UNPACKED_BYTES:
                raise ValueError("Expanded archives and individual artifacts exceed the total size limit")
            destination_file.parent.mkdir(parents=True, exist_ok=True)
            confined(bundle, entry["path"])
            download_archive(entry, destination_file)
        if "data_recipe" in descriptor:
            materialize_source_data(descriptor["data_recipe"], bundle, stage / "official-source.json")
        if local_retriever is not None:
            install_local_retriever(bundle, local_retriever)
        result = verify_bundle(bundle)
        if destination.exists():
            raise ValueError("Destination appeared during download; it was not overwritten")
        bundle.rename(destination)
    return {"reused": False, **result}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "reports" / "artifact-download.json")
    parser.add_argument("--destination", type=Path, default=PROJECT_ROOT / "artifacts" / "runtime")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--local-retriever", type=Path,
                        help="Explicitly supply the hash-verified converted original model.safetensors for its recorded manifest path")
    args = parser.parse_args()
    try:
        if args.verify_only and args.local_retriever is not None:
            raise ValueError("--local-retriever prepares a new bundle and cannot be combined with --verify-only")
        result = verify_bundle(args.destination) if args.verify_only else install_bundle(args.manifest, args.destination,
                                                                                     local_retriever=args.local_retriever)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"Artifact preparation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"destination": str(args.destination.resolve()), **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
