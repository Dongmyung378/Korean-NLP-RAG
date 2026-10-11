"""Install locally, verify the pinned model bundle and run the persistent CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.download_artifacts import install_bundle, verify_bundle
from scripts.setup_runtime import child_environment, setup


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--doctor", action="store_true")
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--setup-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--runtime-venv", type=Path, default=PROJECT_ROOT / ".venv")
    parser.add_argument("--bundle", type=Path, default=PROJECT_ROOT / "artifacts" / "runtime")
    parser.add_argument("--download-manifest", type=Path, default=PROJECT_ROOT / "reports" / "artifact-download.json")
    args, application_args = parser.parse_known_args()
    requested_device = None
    for index, value in enumerate(application_args):
        if value == "--device" and index + 1 < len(application_args):
            requested_device = application_args[index + 1]
        elif value.startswith("--device="):
            requested_device = value.split("=", 1)[1]
    if requested_device == "cuda":
        args.setup_device = "cuda"
    if requested_device is None:
        application_args += ["--device", "cpu"]
    if not any(value in {"--question", "-q", "--interactive", "--questions-file", "--help", "-h"}
               or value.startswith(("--question=", "--questions-file=")) for value in application_args):
        application_args.append("--interactive")
    try:
        python = setup(PROJECT_ROOT, args.runtime_venv, args.setup_device, repair=True)
        if args.setup_only:
            print(json.dumps({"ready": True, "python": str(python)}))
            return 0
        if args.no_download or args.bundle.exists():
            verify_bundle(args.bundle)
        else:
            install_bundle(args.download_manifest, args.bundle)
        env = child_environment(PROJECT_ROOT)
        if args.doctor:
            command = [str(python), str(PROJECT_ROOT / "scripts" / "doctor.py"), "--bundle", str(args.bundle)]
            if args.setup_device == "cuda":
                command.append("--cuda")
        else:
            command = [str(python), "-m", "korean_rag", "--bundle", str(args.bundle), *application_args]
        return subprocess.run(command, cwd=PROJECT_ROOT, env=env).returncode
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"RAG could not start: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
