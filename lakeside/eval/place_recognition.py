"""Cross-season place-recognition benchmark helpers.

This module builds LiDAR-frame place recognition metadata from RTK positions.
It intentionally does not implement learned descriptors. The included dummy
retrieval baseline is only a plumbing check based on sequence index order.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def make_reference_database(
    sequence,
    sensor: str = "lidar",
    min_rtk_status: int = 3,
    origin: tuple[float, float, float] | None = None,
) -> dict:
    """Build a reference database with sensor frames and RTK ENU positions."""

    return _make_sensor_place_set(
        sequence,
        sensor=sensor,
        min_rtk_status=min_rtk_status,
        origin=origin,
        split="reference",
    )


def make_query_set(
    sequence,
    sensor: str = "lidar",
    min_rtk_status: int = 3,
    origin: tuple[float, float, float] | None = None,
) -> dict:
    """Build a query set with sensor frames and RTK ENU positions."""

    return _make_sensor_place_set(
        sequence,
        sensor=sensor,
        min_rtk_status=min_rtk_status,
        origin=origin,
        split="query",
    )


def compute_ground_truth_matches(
    ref_positions, query_positions, distance_threshold: float = 5.0
) -> list[list[int]]:
    """Return positive reference indices for every query position."""

    ref_array = np.asarray(ref_positions, dtype=float)
    query_array = np.asarray(query_positions, dtype=float)
    if ref_array.ndim != 2 or ref_array.shape[1] != 3:
        raise ValueError("Reference positions must have shape (N, 3).")
    if query_array.ndim != 2 or query_array.shape[1] != 3:
        raise ValueError("Query positions must have shape (M, 3).")

    matches: list[list[int]] = []
    for query_position in query_array:
        distances = np.linalg.norm(ref_array - query_position, axis=1)
        positives = np.flatnonzero(distances <= distance_threshold).astype(int)
        matches.append(positives.tolist())
    return matches


def evaluate_retrieval(
    pred_matches, gt_matches, k_values: Iterable[int] = (1, 5, 10)
) -> dict:
    """Evaluate retrieval predictions with Recall@K."""

    predictions = _normalize_predictions(pred_matches)
    ground_truth = _normalize_ground_truth(gt_matches)
    query_indices = [index for index, positives in ground_truth.items() if positives]

    metrics: dict[str, float | int] = {
        "num_queries": len(ground_truth),
        "num_evaluated_queries": len(query_indices),
    }
    for k in k_values:
        hits = 0
        for query_index in query_indices:
            positives = set(ground_truth[query_index])
            retrieved = predictions.get(query_index, [])[:k]
            if positives.intersection(retrieved):
                hits += 1
        metrics[f"Recall@{k}"] = (
            float(hits / len(query_indices)) if query_indices else 0.0
        )
    return metrics


def dummy_index_retrieval(
    num_queries: int, num_refs: int, max_k: int = 10
) -> list[dict[str, list[int] | int]]:
    """Placeholder baseline that retrieves nearest reference sequence indices."""

    predictions = []
    for query_index in range(num_queries):
        center = min(query_index, max(num_refs - 1, 0))
        candidates = sorted(
            range(num_refs), key=lambda ref_index: (abs(ref_index - center), ref_index)
        )
        predictions.append(
            {
                "query_index": query_index,
                "retrieved_ref_indices": candidates[:max_k],
            }
        )
    return predictions


def _make_sensor_place_set(
    sequence,
    sensor: str,
    min_rtk_status: int,
    origin: tuple[float, float, float] | None,
    split: str,
) -> dict:
    if sensor != "lidar":
        raise NotImplementedError("Only sensor='lidar' is supported for now.")

    rtk_rows = sequence.load_metrics().rtk_to_enu(
        min_status=min_rtk_status, origin=origin
    )
    if rtk_rows.empty:
        return _empty_place_set(sequence, sensor=sensor, split=split)

    rtk_timestamps = rtk_rows["timeStamp"].to_numpy(dtype=float)
    rtk_positions = rtk_rows[["enu_x", "enu_y", "enu_z"]].to_numpy(dtype=float)

    indices: list[int] = []
    timestamps: list[float] = []
    positions: list[np.ndarray] = []
    rtk_timestamps_for_frames: list[float] = []
    paths: list[str] = []

    for frame in sequence.iter_lidar():
        rtk_index = _nearest_timestamp_index(rtk_timestamps, frame.timestamp)
        if rtk_index is None:
            continue
        indices.append(frame.index)
        timestamps.append(frame.timestamp)
        positions.append(rtk_positions[rtk_index])
        rtk_timestamps_for_frames.append(float(rtk_timestamps[rtk_index]))
        paths.append(str(frame.path))

    return {
        "split": split,
        "date": sequence.date,
        "traverse": sequence.traverse,
        "sensor": sensor,
        "indices": np.asarray(indices, dtype=int),
        "timestamps": np.asarray(timestamps, dtype=float),
        "positions": np.asarray(positions, dtype=float).reshape((-1, 3)),
        "rtk_timestamps": np.asarray(rtk_timestamps_for_frames, dtype=float),
        "paths": paths,
    }


def _nearest_timestamp_index(timestamps: np.ndarray, query_timestamp: float) -> int | None:
    if timestamps.size == 0:
        return None
    insert_at = int(np.searchsorted(timestamps, query_timestamp))
    if insert_at == 0:
        return 0
    if insert_at == len(timestamps):
        return len(timestamps) - 1
    before = insert_at - 1
    after = insert_at
    before_delta = abs(timestamps[before] - query_timestamp)
    after_delta = abs(timestamps[after] - query_timestamp)
    return before if before_delta <= after_delta else after


def _normalize_predictions(pred_matches) -> dict[int, list[int]]:
    if isinstance(pred_matches, dict) and "predictions" in pred_matches:
        pred_matches = pred_matches["predictions"]
    if isinstance(pred_matches, dict):
        return {
            int(query_index): [int(index) for index in retrieved]
            for query_index, retrieved in pred_matches.items()
        }
    normalized = {}
    for item in pred_matches:
        normalized[int(item["query_index"])] = [
            int(index) for index in item["retrieved_ref_indices"]
        ]
    return normalized


def _normalize_ground_truth(gt_matches) -> dict[int, list[int]]:
    if isinstance(gt_matches, dict) and "pairs" in gt_matches:
        gt_matches = gt_matches["pairs"]
    if isinstance(gt_matches, dict):
        return {
            int(query_index): [int(index) for index in positives]
            for query_index, positives in gt_matches.items()
        }
    normalized = {}
    for query_index, item in enumerate(gt_matches):
        if isinstance(item, dict):
            normalized[int(item["query_index"])] = [
                int(index) for index in item["positive_ref_indices"]
            ]
        else:
            normalized[query_index] = [int(index) for index in item]
    return normalized


def _empty_place_set(sequence, sensor: str, split: str) -> dict:
    return {
        "split": split,
        "date": sequence.date,
        "traverse": sequence.traverse,
        "sensor": sensor,
        "indices": np.asarray([], dtype=int),
        "timestamps": np.asarray([], dtype=float),
        "positions": np.empty((0, 3), dtype=float),
        "rtk_timestamps": np.asarray([], dtype=float),
        "paths": [],
    }
