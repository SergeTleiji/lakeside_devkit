#!/usr/bin/env python3
"""Evaluate a predicted TUM trajectory against ground truth."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lakeside.eval.trajectory import (  # noqa: E402
    associate_by_timestamp,
    compute_position_errors,
    load_tum_trajectory,
    summarize_errors,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare a predicted TUM trajectory to RTK ground truth."
    )
    parser.add_argument("--gt", required=True, type=Path, help="Ground truth TUM file.")
    parser.add_argument("--pred", required=True, type=Path, help="Predicted TUM file.")
    parser.add_argument(
        "--max-dt",
        type=float,
        default=0.05,
        help="Maximum timestamp association difference in seconds.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    gt = load_tum_trajectory(args.gt)
    pred = load_tum_trajectory(args.pred)
    association = associate_by_timestamp(gt, pred, max_dt=args.max_dt)
    errors = compute_position_errors(association["gt_xyz"], association["pred_xyz"])
    summary = summarize_errors(errors)
    print_metrics_table(summary)
    return 0


def print_metrics_table(summary: dict[str, float | int]) -> None:
    rows = [
        ("associated poses", summary["num_associated"]),
        ("RMSE", summary["rmse"]),
        ("mean error", summary["mean"]),
        ("median error", summary["median"]),
        ("95th percentile", summary["p95"]),
        ("max error", summary["max"]),
    ]
    print("metric\tvalue")
    for name, value in rows:
        if isinstance(value, int):
            print(f"{name}\t{value}")
        else:
            print(f"{name}\t{value:.6f}")


if __name__ == "__main__":
    raise SystemExit(main())
