#!/usr/bin/env python3
"""Print RTK status statistics for one Lakeside sequence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lakeside import LakesideSequence  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize RTK status values from a Lakeside Imu.csv file."
    )
    parser.add_argument("--root", required=True, type=Path, help="Dataset root.")
    parser.add_argument("--date", required=True, help="YYMMDD recording date.")
    parser.add_argument(
        "--traverse",
        required=True,
        choices=("on-foot", "scooter"),
        help="Traversal type.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    sequence = LakesideSequence(args.root, args.date, args.traverse)
    summary = sequence.metrics_summary()

    print("Lakeside RTK summary")
    print(f"rows: {summary['rows']}")
    print(
        "timestamp range: "
        f"{summary['timestamp_start']} to {summary['timestamp_end']}"
    )
    print()

    print("RTK status counts and percentages:")
    counts = summary["rtk_status_counts"]
    percentages = summary["rtk_status_percentages"]
    if counts.empty:
        print("  none")
    else:
        for status, count in counts.items():
            percentage = percentages.loc[status]
            print(f"  {status}: {int(count)} ({percentage:.2f}%)")
    print()

    print(f"valid RTK fix rows, status == 3: {summary['rtk_fix_rows']}")
    print(
        "RTK float/fix rows, status >= 2: "
        f"{summary['rtk_float_or_fix_rows']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
