"""Report dependency and artifact health without downloading or loading models."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def inspect_runtime(bundle: Path, *, cuda: bool = False) -> dict:
    problems: list[str] = []
    versions: dict[str, str | None] = {}
    for line in (PROJECT_ROOT / "requirements-runtime.txt").read_text(encoding="utf-8").splitlines() + ["torch==2.14.0"]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==")
        try:
            actual = metadata.version(name)
            if actual.split("+")[0] != expected:
                problems.append(f"{name}: expected {expected}, found {actual}")
        except metadata.PackageNotFoundError:
            actual = None
            problems.append(f"{name}: missing")
        versions[name] = actual
    if sys.version_info[:2] != (3, 12) or sys.maxsize <= 2**32:
        problems.append("64-bit Python 3.12 is required")
    artifact_check = {"path": str(bundle.resolve()), "valid": False}
    try:
        from scripts.download_artifacts import verify_bundle
        artifact_check.update(verify_bundle(bundle))
    except (OSError, ValueError) as error:
        artifact_check["error"] = str(error)
        problems.append("Portable artifact bundle is absent or fails verification")
    if cuda:
        try:
            import torch
            gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            if gpu is None:
                problems.append("CUDA was requested but is unavailable")
        except ImportError:
            gpu = None
    else:
        gpu = None
    return {"healthy": not problems, "python": platform.python_version(), "executable": sys.executable,
            "virtual_environment": sys.prefix != sys.base_prefix, "versions": versions,
            "artifacts": artifact_check, "requested_cuda_device": gpu, "problems": problems}


def main() -> int:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=PROJECT_ROOT / "artifacts" / "runtime")
    parser.add_argument("--cuda", action="store_true", help="Also probe CUDA availability")
    args = parser.parse_args()
    result = inspect_runtime(args.bundle, cuda=args.cuda)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
