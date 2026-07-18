#!/usr/bin/env python3
"""Validate basic Lakeside dataset folder discovery."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lakeside import LakesideDataset  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List discovered Lakeside sequences under a dataset root."
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Dataset root containing Lakeside component folders.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset = LakesideDataset(args.root)
    sequences = dataset.discover_sequences()
    print(f"discovered sequences: {len(sequences)}")
    for sequence in sequences:
        print(f"  {sequence.date} {sequence.traverse}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
