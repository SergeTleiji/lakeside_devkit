#!/usr/bin/env python3
"""Export Lakeside metrics poses in TUM trajectory format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lakeside import LakesideSequence, normalize_quat  # noqa: E402


ORIENTATION_COLUMNS = [
    "Orientation_x",
    "Orientation_y",
    "Orientation_z",
    "Orientation_w",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export RTK or INS trajectory rows in TUM format."
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
        help="Position source to export.",
    )
    parser.add_argument(
        "--min-status",
        type=int,
        default=3,
        help="Minimum RTK_Status for --source rtk. Defaults to 3.",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output TUM txt path.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    sequence = LakesideSequence(args.root, args.date, args.traverse)
    metrics = sequence.load_metrics()
    origin = choose_origin(metrics, source=args.source, min_status=args.min_status)

    if args.source == "rtk":
        dataframe = metrics.rtk_to_enu(min_status=args.min_status, origin=origin)
        allow_identity_orientation = True
    else:
        dataframe = metrics.ins_to_enu(origin=origin)
        allow_identity_orientation = False

    if dataframe.empty:
        raise SystemExit("No rows available for the requested export.")

    use_identity = not all(column in dataframe.columns for column in ORIENTATION_COLUMNS)
    if use_identity and not allow_identity_orientation:
        missing = [column for column in ORIENTATION_COLUMNS if column not in dataframe]
        raise SystemExit(f"Cannot export INS TUM; missing columns: {', '.join(missing)}")
    if use_identity:
        print("Warning: orientation columns are unavailable; using identity quaternion.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    invalid_quaternions = 0
    with args.out.open("w") as stream:
        for _, row in dataframe.iterrows():
            if use_identity:
                qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
            else:
                qx, qy, qz, qw, valid = row_quaternion(row)
                if not valid:
                    invalid_quaternions += 1
            stream.write(
                f"{float(row['timeStamp']):.9f} "
                f"{float(row['enu_x']):.9f} "
                f"{float(row['enu_y']):.9f} "
                f"{float(row['enu_z']):.9f} "
                f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n"
            )

    if invalid_quaternions:
        print(
            "Warning: "
            f"{invalid_quaternions} invalid quaternion rows used identity orientation."
        )
    print(f"exported rows: {len(dataframe)}")
    print(f"output: {args.out}")
    return 0


def row_quaternion(row) -> tuple[float, float, float, float, bool]:
    values = [float(row[column]) for column in ORIENTATION_COLUMNS]
    if not np.all(np.isfinite(values)):
        return 0.0, 0.0, 0.0, 1.0, False
    try:
        qx, qy, qz, qw = normalize_quat(values)
    except ValueError:
        return 0.0, 0.0, 0.0, 1.0, False
    return float(qx), float(qy), float(qz), float(qw), True


def choose_origin(metrics, source: str, min_status: int) -> tuple[float, float, float]:
    rtk_rows = metrics.get_rtk_positions(min_status=min_status).dropna(
        subset=["latRTK", "longRTK", "altRTK"]
    )
    if not rtk_rows.empty:
        row = rtk_rows.iloc[0]
        return float(row["latRTK"]), float(row["longRTK"]), float(row["altRTK"])

    if source == "rtk":
        raise SystemExit("No RTK rows available to define the ENU export origin.")

    ins_rows = metrics.get_ins_positions().dropna(subset=["latINS", "longINS", "altINS"])
    if ins_rows.empty:
        raise SystemExit("No INS rows available to define the ENU export origin.")
    row = ins_rows.iloc[0]
    print("Warning: no RTK origin was available; using the first INS coordinate.")
    return float(row["latINS"]), float(row["longINS"]), float(row["altINS"])


if __name__ == "__main__":
    raise SystemExit(main())
