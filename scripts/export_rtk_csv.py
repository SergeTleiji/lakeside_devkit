#!/usr/bin/env python3
"""Export RTK position rows from one Lakeside sequence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lakeside import LakesideSequence  # noqa: E402


EXPORT_COLUMNS = ["timeStamp", "latRTK", "longRTK", "altRTK", "RTK_Status"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export RTK position columns from a Lakeside Imu.csv file."
    )
    parser.add_argument("--root", required=True, type=Path, help="Dataset root.")
    parser.add_argument("--date", required=True, help="YYMMDD recording date.")
    parser.add_argument(
        "--traverse",
        required=True,
        choices=("on-foot", "scooter"),
        help="Traversal type.",
    )
    parser.add_argument(
        "--min-status",
        type=int,
        default=3,
        help="Minimum RTK_Status to export. Defaults to 3.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output CSV path.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    sequence = LakesideSequence(args.root, args.date, args.traverse)
    metrics = sequence.load_metrics()
    dataframe = metrics.get_rtk_positions(min_status=args.min_status)

    missing = [column for column in EXPORT_COLUMNS if column not in dataframe.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise SystemExit(f"Cannot export RTK CSV; missing columns: {missing_text}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dataframe.loc[:, EXPORT_COLUMNS].to_csv(args.out, index=False)
    print(f"exported rows: {len(dataframe)}")
    print(f"output: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
