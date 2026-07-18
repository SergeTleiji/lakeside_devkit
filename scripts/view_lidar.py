#!/usr/bin/env python3
"""Plot a simple XY scatter view of one LiDAR PCD frame."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lakeside import LakesideSequence, load_pcd_xyz  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load a PCD frame and save a simple XY scatter plot."
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
        "--index",
        required=True,
        type=int,
        help="Zero-based sorted LiDAR frame index.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=100000,
        help="Maximum points to plot after deterministic stride sampling.",
    )
    parser.add_argument(
        "--pcd-backend",
        choices=("auto", "ascii", "open3d"),
        default="auto",
        help="PCD loading backend. Defaults to auto.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to outputs/{date}_{traverse}_lidar_{index}.png.",
    )
    parser.add_argument("--show", action="store_true", help="Display the plot window.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    plt = import_pyplot(show=args.show)

    sequence = LakesideSequence(args.root, args.date, args.traverse)
    try:
        frame = sequence.get_lidar(args.index)
    except IndexError as error:
        raise SystemExit(f"No LiDAR frame at sorted index {args.index}.") from error

    try:
        points = load_pcd_xyz(frame.path, backend=args.pcd_backend)
    except NotImplementedError as error:
        print(error)
        print(
            "Open3D is required for this PCD format. Install with: pip install open3d"
        )
        return 1
    except ImportError as error:
        print(error)
        return 1

    if points.size == 0:
        raise SystemExit(f"No points found in {frame.path}")
    if args.max_points > 0 and len(points) > args.max_points:
        step = max(len(points) // args.max_points, 1)
        points = points[::step]

    figure, axis = plt.subplots(figsize=(7, 7))
    axis.scatter(points[:, 0], points[:, 1], s=0.2)
    axis.set_title(f"{args.date} {args.traverse} LiDAR {args.index}")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.axis("equal")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()

    out_path = args.out or Path("outputs") / (
        f"{args.date}_{args.traverse}_lidar_{args.index}.png"
    )
    save_figure(figure, out_path)
    print(f"saved: {out_path}")

    if args.show:
        plt.show()
    plt.close(figure)
    return 0


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
