#!/usr/bin/env python3
"""Evaluate INS positions against RTK ground truth for one Lakeside sequence."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lakeside import LakesideSequence  # noqa: E402
from lakeside.eval.trajectory import (  # noqa: E402
    associate_by_timestamp,
    compute_position_errors,
    summarize_errors,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the first Lakeside sanity benchmark: compare INS positions to "
            "RTK ground truth after timestamp association."
        )
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
        help="Minimum RTK_Status for ground truth rows. Defaults to 3.",
    )
    parser.add_argument(
        "--max-dt",
        type=float,
        default=0.05,
        help="Maximum timestamp association difference in seconds.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help=(
            "Output JSON path. Defaults to "
            "outputs/{date}_{traverse}_ins_vs_rtk_metrics.json."
        ),
    )
    parser.add_argument(
        "--out-plot",
        type=Path,
        default=None,
        help=(
            "Output plot path. Defaults to "
            "outputs/{date}_{traverse}_ins_vs_rtk.png."
        ),
    )
    parser.add_argument("--show", action="store_true", help="Display the plot window.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    plt = import_pyplot(show=args.show)

    sequence = LakesideSequence(args.root, args.date, args.traverse)
    metrics = sequence.load_metrics()

    rtk_rows = metrics.get_rtk_positions(min_status=args.min_status).dropna(
        subset=["latRTK", "longRTK", "altRTK"]
    )
    if rtk_rows.empty:
        raise SystemExit("No RTK rows available for the requested minimum status.")
    origin = (
        float(rtk_rows.iloc[0]["latRTK"]),
        float(rtk_rows.iloc[0]["longRTK"]),
        float(rtk_rows.iloc[0]["altRTK"]),
    )

    rtk = metrics.rtk_to_enu(min_status=args.min_status, origin=origin)
    ins = metrics.ins_to_enu(origin=origin)
    gt = dataframe_to_trajectory(rtk)
    pred = dataframe_to_trajectory(ins)

    association = associate_by_timestamp(gt, pred, max_dt=args.max_dt)
    errors = compute_position_errors(association["gt_xyz"], association["pred_xyz"])
    summary = summarize_errors(errors)
    result = {
        "date": args.date,
        "traverse": args.traverse,
        "benchmark": "ins_vs_rtk",
        "rtk_min_status": args.min_status,
        "max_dt": args.max_dt,
        "origin": {
            "lat": origin[0],
            "lon": origin[1],
            "alt": origin[2],
        },
        **summary,
    }

    out_json = args.out_json or Path("outputs") / (
        f"{args.date}_{args.traverse}_ins_vs_rtk_metrics.json"
    )
    out_plot = args.out_plot or Path("outputs") / (
        f"{args.date}_{args.traverse}_ins_vs_rtk.png"
    )
    save_json(result, out_json)
    save_plot(plt, rtk, ins, association, out_plot)

    print_metrics_table(summary)
    print(f"metrics_json\t{out_json}")
    print(f"plot\t{out_plot}")
    if args.show:
        plt.show()
    return 0


def dataframe_to_trajectory(dataframe) -> dict:
    return {
        "timestamps": dataframe["timeStamp"].to_numpy(dtype=float),
        "xyz": dataframe[["enu_x", "enu_y", "enu_z"]].to_numpy(dtype=float),
    }


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


def save_json(result: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


def save_plot(plt, rtk, ins, association: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 7))
    axis.plot(rtk["enu_x"], rtk["enu_y"], label="RTK")
    axis.plot(ins["enu_x"], ins["enu_y"], label="INS")
    if len(association["gt_xyz"]) > 0:
        axis.scatter(
            association["gt_xyz"][:, 0],
            association["gt_xyz"][:, 1],
            s=4,
            label="associated RTK",
        )
    axis.set_title("INS vs RTK trajectory")
    axis.set_xlabel("east [m]")
    axis.set_ylabel("north [m]")
    axis.axis("equal")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


def import_pyplot(show: bool):
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lakeside_matplotlib")
    )
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


if __name__ == "__main__":
    raise SystemExit(main())
