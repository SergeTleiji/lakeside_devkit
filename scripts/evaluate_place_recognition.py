#!/usr/bin/env python3
"""Evaluate place-recognition retrieval predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lakeside.eval.place_recognition import evaluate_retrieval  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate cross-season place-recognition predictions."
    )
    parser.add_argument("--pairs", required=True, type=Path, help="Pair JSON path.")
    parser.add_argument(
        "--pred",
        required=True,
        type=Path,
        help="Retrieval prediction JSON path.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pairs = read_json(args.pairs)
    predictions = read_json(args.pred)
    metrics = evaluate_retrieval(
        predictions,
        pairs,
        k_values=(1, 5, 10),
    )
    print_metrics(metrics)
    return 0


def read_json(path: Path) -> dict:
    with path.open("r") as stream:
        return json.load(stream)


def print_metrics(metrics: dict[str, float | int]) -> None:
    print("metric\tvalue")
    print(f"num_queries\t{metrics['num_queries']}")
    print(f"num_evaluated_queries\t{metrics['num_evaluated_queries']}")
    for key in ("Recall@1", "Recall@5", "Recall@10"):
        print(f"{key}\t{metrics[key]:.6f}")


if __name__ == "__main__":
    raise SystemExit(main())
