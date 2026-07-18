#!/usr/bin/env python3
"""Plot RTK or INS trajectory points from a Lakeside Imu.csv file."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lakeside import LakesideSequence  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot a simple RTK or INS trajectory from Lakeside metrics."
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
        "--source",
        required=True,
        choices=("rtk", "ins"),
        help="Trajectory source to plot.",
    )
    parser.add_argument(
        "--min-status",
        type=int,
        default=None,
        help="For RTK plots, keep rows with RTK_Status at least this value.",
    )
    parser.add_argument(
        "--color-by-status",
        action="store_true",
        help="For RTK plots, color points by RTK_Status.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to outputs/{date}_{traverse}_trajectory.png.",
    )
    parser.add_argument("--show", action="store_true", help="Display the plot window.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    plt = import_pyplot(show=args.show)

    sequence = LakesideSequence(args.root, args.date, args.traverse)
    metrics = sequence.load_metrics()
    dataframe, x_column, y_column = trajectory_dataframe(
        metrics.dataframe,
        source=args.source,
        min_status=args.min_status,
    )
    if dataframe.empty:
        raise SystemExit("No trajectory rows available for the requested options.")

    figure, axis = plt.subplots(figsize=(8, 6))
    if args.source == "rtk" and args.color_by_status and "RTK_Status" in dataframe:
        scatter = axis.scatter(
            dataframe[x_column],
            dataframe[y_column],
            s=4,
            c=dataframe["RTK_Status"],
        )
        figure.colorbar(scatter, ax=axis, label="RTK_Status")
    else:
        axis.scatter(dataframe[x_column], dataframe[y_column], s=4)

    axis.set_title(f"{args.date} {args.traverse} {args.source.upper()} trajectory")
    axis.set_xlabel(x_column)
    axis.set_ylabel(y_column)
    axis.axis("equal")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()

    out_path = args.out or Path("outputs") / f"{args.date}_{args.traverse}_trajectory.png"
    save_figure(figure, out_path)
    print(f"saved: {out_path}")

    if args.show:
        plt.show()
    plt.close(figure)
    return 0


def trajectory_dataframe(dataframe, source: str, min_status: int | None):
    if source == "rtk":
        required = ["longRTK", "latRTK"]
        missing = [column for column in required if column not in dataframe]
        if missing:
            raise SystemExit(f"Missing RTK trajectory columns: {', '.join(missing)}")
        result = dataframe.dropna(subset=required).copy()
        if min_status is not None:
            if "RTK_Status" not in result:
                raise SystemExit("Cannot filter RTK trajectory; RTK_Status is missing.")
            result = result.loc[result["RTK_Status"] >= min_status]
        return result, "longRTK", "latRTK"

    if {"INS_x", "INS_y"}.issubset(dataframe.columns):
        columns = ["INS_x", "INS_y"]
        return dataframe.dropna(subset=columns).copy(), "INS_x", "INS_y"

    required = ["longINS", "latINS"]
    missing = [column for column in required if column not in dataframe]
    if missing:
        raise SystemExit(f"Missing INS trajectory columns: {', '.join(missing)}")
    return dataframe.dropna(subset=required).copy(), "longINS", "latINS"


def import_pyplot(show: bool):
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lakeside_matplotlib")
    )
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_figure(figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)


if __name__ == "__main__":
    raise SystemExit(main())
