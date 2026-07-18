"""Trajectory loading, association, and position error metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_tum_trajectory(path: str | Path) -> dict[str, np.ndarray]:
    """Load a TUM trajectory text file.

    Expected columns are ``timestamp tx ty tz qx qy qz qw``. Comment and blank
    lines are ignored.
    """

    timestamps: list[float] = []
    positions: list[tuple[float, float, float]] = []
    quaternions: list[tuple[float, float, float, float]] = []

    with Path(path).expanduser().open("r") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 8:
                raise ValueError(
                    f"Expected at least 8 TUM columns at {path}:{line_number}"
                )
            values = [float(value) for value in parts[:8]]
            timestamps.append(values[0])
            positions.append((values[1], values[2], values[3]))
            quaternions.append((values[4], values[5], values[6], values[7]))

    return {
        "timestamps": np.asarray(timestamps, dtype=float),
        "xyz": np.asarray(positions, dtype=float),
        "quat": np.asarray(quaternions, dtype=float),
    }


def associate_by_timestamp(
    gt: dict[str, np.ndarray], pred: dict[str, np.ndarray], max_dt: float = 0.05
) -> dict[str, np.ndarray]:
    """Associate predicted poses to ground truth poses by nearest timestamp."""

    gt_timestamps = np.asarray(gt["timestamps"], dtype=float)
    pred_timestamps = np.asarray(pred["timestamps"], dtype=float)
    gt_xyz = np.asarray(gt["xyz"], dtype=float)
    pred_xyz = np.asarray(pred["xyz"], dtype=float)

    if gt_timestamps.size == 0 or pred_timestamps.size == 0:
        return _empty_association()

    order = np.argsort(gt_timestamps)
    gt_timestamps = gt_timestamps[order]
    gt_xyz = gt_xyz[order]

    gt_indices: list[int] = []
    pred_indices: list[int] = []
    deltas: list[float] = []

    for pred_index, pred_timestamp in enumerate(pred_timestamps):
        insert_at = np.searchsorted(gt_timestamps, pred_timestamp)
        candidates = []
        if insert_at > 0:
            candidates.append(insert_at - 1)
        if insert_at < len(gt_timestamps):
            candidates.append(insert_at)
        if not candidates:
            continue

        best_gt_index = min(
            candidates,
            key=lambda gt_index: abs(gt_timestamps[gt_index] - pred_timestamp),
        )
        delta = float(pred_timestamp - gt_timestamps[best_gt_index])
        if abs(delta) <= max_dt:
            gt_indices.append(int(best_gt_index))
            pred_indices.append(pred_index)
            deltas.append(delta)

    if not gt_indices:
        return _empty_association()

    gt_index_array = np.asarray(gt_indices, dtype=int)
    pred_index_array = np.asarray(pred_indices, dtype=int)
    return {
        "gt_timestamps": gt_timestamps[gt_index_array],
        "pred_timestamps": pred_timestamps[pred_index_array],
        "gt_xyz": gt_xyz[gt_index_array],
        "pred_xyz": pred_xyz[pred_index_array],
        "dt": np.asarray(deltas, dtype=float),
        "gt_indices": gt_index_array,
        "pred_indices": pred_index_array,
    }


def compute_position_errors(gt_xyz, pred_xyz) -> np.ndarray:
    """Return Euclidean position errors for associated poses."""

    gt_array = np.asarray(gt_xyz, dtype=float)
    pred_array = np.asarray(pred_xyz, dtype=float)
    if gt_array.shape != pred_array.shape:
        raise ValueError("Ground truth and prediction arrays must have the same shape.")
    if gt_array.ndim != 2 or gt_array.shape[1] != 3:
        raise ValueError("Position arrays must have shape (N, 3).")
    return np.linalg.norm(gt_array - pred_array, axis=1)


def compute_ate(gt_xyz, pred_xyz) -> float:
    """Return absolute trajectory error RMSE for associated positions."""

    errors = compute_position_errors(gt_xyz, pred_xyz)
    if errors.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(errors**2)))


def summarize_errors(errors) -> dict[str, float | int]:
    """Return common position error summary metrics."""

    error_array = np.asarray(errors, dtype=float)
    if error_array.size == 0:
        return {
            "num_associated": 0,
            "rmse": float("nan"),
            "mean": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }

    return {
        "num_associated": int(error_array.size),
        "rmse": float(np.sqrt(np.mean(error_array**2))),
        "mean": float(np.mean(error_array)),
        "median": float(np.median(error_array)),
        "p95": float(np.percentile(error_array, 95)),
        "max": float(np.max(error_array)),
    }


def _empty_association() -> dict[str, np.ndarray]:
    return {
        "gt_timestamps": np.asarray([], dtype=float),
        "pred_timestamps": np.asarray([], dtype=float),
        "gt_xyz": np.empty((0, 3), dtype=float),
        "pred_xyz": np.empty((0, 3), dtype=float),
        "dt": np.asarray([], dtype=float),
        "gt_indices": np.asarray([], dtype=int),
        "pred_indices": np.asarray([], dtype=int),
    }
