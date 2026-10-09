"""Nyquist validation tests for Phase 1 (Runnable Toolchain).

Requirements covered: ENV-01, ENV-02, ENV-03, ENV-04, CFG-03.
"""

import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
README = ROOT / "README.md"
CLI = ROOT / "src" / "imagai" / "cli.py"
LOCK = ROOT / "uv.lock"
PYTHON_VERSION = ROOT / ".python-version"

_SCAN_SKIP_DIRS = {
    ".git",
    ".planning",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "graft",
}


def test_lock_is_in_sync_with_pyproject_toml():
    """ENV-01: the documented uv workflow resolves a lockfile matching pyproject.toml."""
    if shutil.which("uv") is None:
        import pytest

        pytest.skip("uv not on PATH — cannot verify lock consistency")
    result = subprocess.run(
        ["uv", "lock", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"uv lock --check failed:\n{result.stdout}\n{result.stderr}"


def test_readme_documents_single_uv_command_workflow():
    """ENV-01: README documents install + test with uv commands only."""
    text = README.read_text()
    assert "uv sync" in text, "README must document 'uv sync' for install"
    assert "uv run pytest" in text, "README must document 'uv run pytest' for tests"
    assert "rye " not in text, "README must not document rye commands"


def test_python_version_consistent_across_pin_lock_and_floor():
    """ENV-02: .python-version satisfies requires-python, and the lock agrees with pyproject."""
    pinned = PYTHON_VERSION.read_text().strip()
    assert re.fullmatch(r"\d+\.\d+(\.\d+)?", pinned), f".python-version not a version: {pinned!r}"

    pyproject_text = PYPROJECT.read_text()
    floor_match = re.search(r'requires-python\s*=\s*">=(\d+)\.(\d+)"', pyproject_text)
    assert floor_match, "pyproject.toml must declare requires-python = '>=X.Y'"
    floor = (int(floor_match.group(1)), int(floor_match.group(2)))

    lock_text = LOCK.read_text()
    assert re.search(r"requires-python\s*=\s*['\"]>=\d+\.\d+['\"]", lock_text), (
        "uv.lock must record a requires-python floor matching pyproject.toml"
    )

    pin_major, pin_minor = (int(p) for p in pinned.split(".")[:2])
    assert (pin_major, pin_minor) >= floor, (
        f".python-version {pinned} does not satisfy pyproject floor {floor}"
    )


def test_no_rye_references_outside_planning():
    """ENV-03: zero rye references in build/docs surface (outside .planning/ and .git)."""
    offenders = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in _SCAN_SKIP_DIRS for part in path.relative_to(ROOT).parts):
            continue
        if path.name == Path(__file__).name:
            continue  # this test file mentions rye in its own assertions
        try:
            if "rye" in path.read_text(errors="strict"):
                offenders.append(str(path))
        except (UnicodeDecodeError, ValueError):
            continue  # binary file
    assert not offenders, f"rye references remain in: {offenders}"


def test_cli_imports_annotated_from_stdlib_typing():
    """ENV-04: cli.py imports Annotated from stdlib typing, not typing_extensions."""
    source = CLI.read_text()
    assert "from typing import Annotated" in source, (
        "cli.py must import Annotated from stdlib typing"
    )
    assert "typing_extensions" not in source, (
        "cli.py must not import typing_extensions (stdlib typing.Annotated is the floor)"
    )


def test_requires_python_floor_is_at_least_39():
    """CFG-03: requires-python states the real >=3.9 floor."""
    floor_match = re.search(r'requires-python\s*=\s*">=(\d+)\.(\d+)"', PYPROJECT.read_text())
    assert floor_match, "pyproject.toml must declare requires-python"
    major, minor = int(floor_match.group(1)), int(floor_match.group(2))
    assert (major, minor) >= (3, 9), f"requires-python floor {major}.{minor} is below 3.9"
