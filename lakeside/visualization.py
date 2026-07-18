"""Visualization helpers that avoid heavy optional dependencies by default."""

from __future__ import annotations

from pathlib import Path


def describe_file(path: Path | None) -> str:
    """Return a short display string for a dataset file path."""

    return str(path) if path is not None else "not found"
