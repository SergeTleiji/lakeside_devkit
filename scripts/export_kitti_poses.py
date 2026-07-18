#!/usr/bin/env python3
"""Export Lakeside INS poses in KITTI 3x4 matrix format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lakeside import LakesideSequence, make_transform, quat_to_rotmat  # noqa: E402


ORIENTATION_COLUMNS = [
    "Orientation_x",
    "Orientation_y",
    "Orientation_z",
    "Orientation_w",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export INS poses in KITTI 3x4 pose matrix format."
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
        choices=("ins",),
        help="Position source to export. Currently only ins is supported.",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output poses.txt path.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    sequence = LakesideSequence(args.root, args.date, args.traverse)
    dataframe = sequence.load_metrics().ins_to_enu()

    missing = [column for column in ORIENTATION_COLUMNS if column not in dataframe]
    if missing:
        raise SystemExit(f"Cannot export KITTI poses; missing columns: {', '.join(missing)}")
    if dataframe.empty:
        raise SystemExit("No INS rows available for KITTI export.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    invalid_quaternions = 0
    with args.out.open("w") as stream:
        for _, row in dataframe.iterrows():
            try:
                rotation = quat_to_rotmat(
                    float(row["Orientation_x"]),
                    float(row["Orientation_y"]),
                    float(row["Orientation_z"]),
                    float(row["Orientation_w"]),
                )
            except ValueError:
                rotation = np.eye(3)
                invalid_quaternions += 1
            translation = np.asarray(
                [float(row["enu_x"]), float(row["enu_y"]), float(row["enu_z"])],
                dtype=float,
            )
            pose = make_transform(rotation, translation)[:3, :]
            values = pose.reshape(-1)
            stream.write(" ".join(f"{value:.9f}" for value in values) + "\n")

    if invalid_quaternions:
        print(
            "Warning: "
            f"{invalid_quaternions} invalid quaternion rows used identity rotation."
        )
    print(f"exported rows: {len(dataframe)}")
    print(f"output: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
