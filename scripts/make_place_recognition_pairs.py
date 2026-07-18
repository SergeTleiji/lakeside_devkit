#!/usr/bin/env python3
"""Generate RTK ground-truth pairs for cross-season place recognition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lakeside import LakesideSequence  # noqa: E402
from lakeside.eval.place_recognition import (  # noqa: E402
    compute_ground_truth_matches,
    dummy_index_retrieval,
    make_query_set,
    make_reference_database,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build LiDAR/RTK ground-truth positive pairs for cross-season "
            "place recognition."
        )
    )
    parser.add_argument("--root", required=True, type=Path, help="Dataset root.")
    parser.add_argument("--ref-date", required=True, help="Reference sequence date.")
    parser.add_argument("--query-date", required=True, help="Query sequence date.")
    parser.add_argument(
        "--traverse",
        required=True,
        choices=("on-foot", "scooter"),
        help="Traversal type for both sequences.",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=5.0,
        help="Positive match radius in meters. Defaults to 5.0.",
    )
    parser.add_argument(
        "--min-rtk-status",
        type=int,
        default=3,
        help="Minimum RTK_Status used for positions. Defaults to 3.",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output JSON path.")
    parser.add_argument(
        "--dummy-pred-out",
        type=Path,
        default=None,
        help="Optional output path for placeholder index-order predictions.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    ref_sequence = LakesideSequence(args.root, args.ref_date, args.traverse)
    query_sequence = LakesideSequence(args.root, args.query_date, args.traverse)
    origin = choose_rtk_origin(ref_sequence, min_status=args.min_rtk_status)

    reference = make_reference_database(
        ref_sequence,
        sensor="lidar",
        min_rtk_status=args.min_rtk_status,
        origin=origin,
    )
    query = make_query_set(
        query_sequence,
        sensor="lidar",
        min_rtk_status=args.min_rtk_status,
        origin=origin,
    )
    if len(reference["positions"]) == 0:
        raise SystemExit("No reference LiDAR/RTK frames were available.")
    if len(query["positions"]) == 0:
        raise SystemExit("No query LiDAR/RTK frames were available.")

    match_rows = compute_ground_truth_matches(
        reference["positions"],
        query["positions"],
        distance_threshold=args.distance_threshold,
    )
    pairs = []
    for query_row, positive_rows in enumerate(match_rows):
        positive_ref_indices = [
            int(reference["indices"][ref_row]) for ref_row in positive_rows
        ]
        positive_ref_timestamps = [
            float(reference["timestamps"][ref_row]) for ref_row in positive_rows
        ]
        pairs.append(
            {
                "query_index": int(query["indices"][query_row]),
                "query_timestamp": float(query["timestamps"][query_row]),
                "positive_ref_indices": positive_ref_indices,
                "positive_ref_timestamps": positive_ref_timestamps,
            }
        )

    output = {
        "ref_date": args.ref_date,
        "query_date": args.query_date,
        "traverse": args.traverse,
        "sensor": "lidar",
        "min_rtk_status": args.min_rtk_status,
        "distance_threshold_m": args.distance_threshold,
        "origin": {
            "lat": origin[0],
            "lon": origin[1],
            "alt": origin[2],
        },
        "num_reference_frames": int(len(reference["indices"])),
        "num_query_frames": int(len(query["indices"])),
        "pairs": pairs,
    }
    write_json(output, args.out)
    print(f"reference frames: {len(reference['indices'])}")
    print(f"query frames: {len(query['indices'])}")
    print(f"queries with positives: {sum(bool(pair['positive_ref_indices']) for pair in pairs)}")
    print(f"output: {args.out}")

    if args.dummy_pred_out is not None:
        predictions = remap_dummy_predictions(
            dummy_index_retrieval(
                num_queries=len(query["indices"]),
                num_refs=len(reference["indices"]),
                max_k=10,
            ),
            query_indices=query["indices"],
            ref_indices=reference["indices"],
        )
        write_json(
            {
                "baseline": "dummy_index_order_placeholder",
                "predictions": predictions,
            },
            args.dummy_pred_out,
        )
        print(f"dummy predictions: {args.dummy_pred_out}")
    return 0


def choose_rtk_origin(
    sequence: LakesideSequence, min_status: int
) -> tuple[float, float, float]:
    rtk_rows = sequence.load_metrics().get_rtk_positions(min_status=min_status).dropna(
        subset=["latRTK", "longRTK", "altRTK"]
    )
    if rtk_rows.empty:
        raise SystemExit("No valid RTK rows found in the reference sequence.")
    row = rtk_rows.iloc[0]
    return float(row["latRTK"]), float(row["longRTK"]), float(row["altRTK"])


def remap_dummy_predictions(predictions, query_indices, ref_indices):
    remapped = []
    for prediction in predictions:
        query_row = int(prediction["query_index"])
        ref_rows = prediction["retrieved_ref_indices"]
        remapped.append(
            {
                "query_index": int(query_indices[query_row]),
                "retrieved_ref_indices": [int(ref_indices[row]) for row in ref_rows],
            }
        )
    return remapped


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
