"""Prepare a project-only Python environment; never install into global Python."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTORCH_VERSION = "2.14.0"
INDEXES = {"cpu": "https://download.pytorch.org/whl/cpu", "cuda": "https://download.pytorch.org/whl/cu130"}
PROBE = r'''
import importlib.metadata as m, json, sys, pathlib
from packaging.version import Version
requirements = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
problems = []
for requirement in requirements + ["torch==2.14.0", "korean-nlp-rag==2.0.0"]:
    requirement = requirement.strip()
    if not requirement or requirement.startswith("#"):
        continue
    package, wanted = requirement.split("==")
    try:
        actual = m.version(package)
        if Version(actual).public != wanted:
            problems.append(f"{package}: wanted {wanted}, found {actual}")
    except m.PackageNotFoundError:
        problems.append(f"{package}: not installed")
if sys.version_info[:2] != (3,12) or sys.maxsize <= 2**32 or sys.prefix == sys.base_prefix:
    problems.append("A Python 3.12 virtual environment is required")
print(json.dumps({"healthy": not problems, "problems": problems}))
sys.exit(bool(problems))
'''


def child_environment(root: Path) -> dict[str, str]:
    """Remove Python/pip redirections while preserving proxy/certificate settings."""
    env = os.environ.copy()
    for name in (
        "PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE", "VIRTUAL_ENV",
        "PIP_TARGET", "PIP_PREFIX", "PIP_ROOT", "PIP_USER", "PIP_CONSTRAINT",
        "PIP_REQUIREMENT", "PIP_NO_DEPS", "PIP_CONFIG_FILE", "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL", "PIP_FIND_LINKS",
    ):
        env.pop(name, None)
    for name in list(env):
        if name.startswith("PIP_") and name not in {"PIP_CERT", "PIP_CLIENT_CERT", "PIP_PROXY", "PIP_TIMEOUT", "PIP_DEFAULT_TIMEOUT", "PIP_RETRIES"}:
            env.pop(name, None)
    temp = root / ".local" / "temp"
    require_inside(root, temp)
    temp.mkdir(parents=True, exist_ok=True)
    require_inside(root, temp)
    env.update(TEMP=str(temp), TMP=str(temp), TMPDIR=str(temp), PYTHONNOUSERSITE="1", PYTHONUTF8="1", PIP_CONFIG_FILE=os.devnull)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(name, "4")
    return env


def require_inside(root: Path, target: Path) -> Path:
    resolved_root = root.resolve()
    resolved = target.resolve()
    if resolved == resolved_root or not resolved.is_relative_to(resolved_root):
        raise ValueError(f"Path must be a child of the project folder: {target}")
    return resolved


def environment_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def healthy(venv: Path, root: Path, env: dict[str, str]) -> bool:
    python = environment_python(venv)
    if not python.is_file():
        return False
    configuration = venv / "pyvenv.cfg"
    if not configuration.is_file() or "include-system-site-packages = true" in configuration.read_text(encoding="utf-8").lower():
        return False
    try:
        result = subprocess.run([str(python), "-c", PROBE, str(root / "requirements-runtime.txt")],
                                cwd=root, env=env, text=True, capture_output=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def setup(root: Path, venv: Path, device: str, *, repair: bool = False) -> Path:
    root = root.resolve()
    venv = require_inside(root, venv)
    if sys.version_info[:2] != (3, 12) or sys.maxsize <= 2**32:
        raise ValueError("Run setup with 64-bit Python 3.12. The platform launcher can obtain it automatically.")
    env = child_environment(root)
    if healthy(venv, root, env):
        python = environment_python(venv)
        if device == "cuda":
            result = subprocess.run([str(python), "-c", "import torch,sys; sys.exit(not torch.cuda.is_available())"], cwd=root, env=env)
            if result.returncode == 0:
                return python
        else:
            return python
    if venv.exists():
        if not repair:
            raise ValueError("The existing environment is incomplete or incompatible. Use --repair to preserve it and create a clean one.")
        backups = root / ".local" / "environment-backups"
        require_inside(root, backups)
        backups.mkdir(parents=True, exist_ok=True)
        backup = require_inside(root, backups / (datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]))
        require_inside(root, venv)
        venv.rename(backup)
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True, cwd=root, env=env)
    python = environment_python(venv)
    pip = [str(python), "-m", "pip", "--disable-pip-version-check", "--no-cache-dir"]
    subprocess.run(pip + ["install", "--index-url", "https://pypi.org/simple", "-r", str(root / "requirements-runtime.txt")],
                   check=True, cwd=root, env=env)
    torch_install = pip + ["install", "--index-url", INDEXES[device], f"torch=={PYTORCH_VERSION}"]
    if device == "cpu":
        torch_install.append("--no-deps")
    subprocess.run(torch_install, check=True, cwd=root, env=env)
    subprocess.run(pip + ["install", "--no-deps", "--no-build-isolation", "-e", str(root)], check=True, cwd=root, env=env)
    subprocess.run([str(python), "-m", "pip", "check"], check=True, cwd=root, env=env)
    if not healthy(venv, root, env):
        raise RuntimeError("Installation finished but the runtime version check failed.")
    if device == "cuda":
        subprocess.run([str(python), "-c", "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'"],
                       check=True, cwd=root, env=env)
    return python


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=PROJECT_ROOT / ".venv")
    parser.add_argument("--device", choices=tuple(INDEXES), default="cpu")
    parser.add_argument("--repair", action="store_true")
    args = parser.parse_args()
    try:
        python = setup(PROJECT_ROOT, args.venv, args.device, repair=args.repair)
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Runtime setup failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"ready": True, "python": str(python), "device": args.device}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

