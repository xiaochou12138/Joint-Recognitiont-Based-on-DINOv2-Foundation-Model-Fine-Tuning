"""Lightweight repository smoke test.

This script is intentionally dependency-light so editors, reviewers, and users
can check the public source tree without downloading seismic data or training a
model.
"""

from __future__ import annotations

import py_compile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = [
    "README.md",
    "LICENSE",
    "requirements.txt",
    "dataset.py",
    "demo_classification.py",
    "demo_classification_two_tasks.py",
    "evaluate_classification.py",
    "evaluate_classification_two_tasks.py",
    "models",
    "loss",
    "data",
    "docs/USAGE.md",
    "docs/COMPUTER_CODE_AVAILABILITY.md",
]


def check_required_paths() -> None:
    missing = [path for path in REQUIRED_PATHS if not (ROOT / path).exists()]
    if missing:
        raise AssertionError(f"Missing required repository paths: {missing}")


def check_requirements_have_unique_names() -> None:
    requirements = ROOT / "requirements.txt"
    names: list[str] = []
    for raw_line in requirements.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name = line
        for separator in ("==", ">=", "<=", "~=", "!=", ">", "<"):
            if separator in line:
                name = line.split(separator, 1)[0]
                break
        names.append(name.lower().replace("_", "-"))

    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise AssertionError(f"Duplicate requirement entries found: {duplicates}")


def check_python_files_compile() -> None:
    skipped = {ROOT / "tests" / "quick_test.py"}
    for source_file in sorted(ROOT.rglob("*.py")):
        if source_file in skipped:
            continue
        if "__pycache__" in source_file.parts:
            continue
        py_compile.compile(str(source_file), doraise=True)


def main() -> None:
    check_required_paths()
    check_requirements_have_unique_names()
    check_python_files_compile()
    print("Quick test passed: repository structure, requirements, and Python syntax are valid.")


if __name__ == "__main__":
    main()
