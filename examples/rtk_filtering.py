"""RTK status filtering example."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lakeside import LakesideSequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter Lakeside RTK metrics rows.")
    parser.add_argument("--root", required=True, type=Path, help="Dataset root.")
    parser.add_argument("--date", default="200803", help="YYMMDD recording date.")
    parser.add_argument(
        "--traverse",
        default="on-foot",
        choices=("on-foot", "scooter"),
        help="Traversal type.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/rtk_fix_rows.csv"),
        help="Output CSV for RTK fix rows.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metrics = LakesideSequence(args.root, args.date, args.traverse).load_metrics()

    print("RTK status counts:")
    print(metrics.rtk_status_counts())
    print()
    print("RTK status percentages:")
    print(metrics.rtk_status_percentages())

    fix_rows = metrics.get_rtk_positions(min_status=3)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fix_rows.to_csv(args.out, index=False)
    print(f"exported RTK fix rows: {len(fix_rows)}")
    print(f"output: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
