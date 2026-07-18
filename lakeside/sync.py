"""Timestamp synchronization helpers.

LiDAR filenames and metrics rows both expose Unix timestamps, so they can be
matched by nearest time. Camera image filenames currently expose sequential
frame IDs only, so camera traversal remains frame-order based.

TODO: Add camera-to-metrics synchronization only if camera timestamp files are
discovered and their format is documented.
"""

from __future__ import annotations

from bisect import bisect_left
from itertools import islice
from typing import TYPE_CHECKING, Any, Iterable

from .sensors import LidarFrame

if TYPE_CHECKING:
    from .sequence import LakesideSequence


def nearest_index(sorted_values: Iterable[float], query_value: float) -> int | None:
    """Return the index of the value nearest to ``query_value``.

    ``sorted_values`` must be sorted in ascending order.
    """

    values = list(sorted_values)
    if not values:
        return None

    insertion_index = bisect_left(values, query_value)
    if insertion_index == 0:
        return 0
    if insertion_index == len(values):
        return len(values) - 1

    before_index = insertion_index - 1
    after_index = insertion_index
    before_delta = abs(values[before_index] - query_value)
    after_delta = abs(values[after_index] - query_value)
    return before_index if before_delta <= after_delta else after_index


def nearest_timestamp(
    sorted_timestamps: Iterable[float], query_timestamp: float
) -> int | None:
    """Return the index of the timestamp nearest to ``query_timestamp``."""

    return nearest_index(sorted_timestamps, query_timestamp)


def match_lidar_to_metrics(
    sequence: "LakesideSequence", lidar_frame: LidarFrame
) -> dict[str, Any] | None:
    """Match one lidar frame to the nearest metrics row by Unix timestamp."""

    metrics = sequence.load_metrics()
    metric_timestamps = metrics.timestamps().tolist()
    metrics_index = nearest_timestamp(metric_timestamps, lidar_frame.timestamp)
    if metrics_index is None:
        return None

    metrics_row = metrics.dataframe.iloc[metrics_index]
    metrics_timestamp = float(metrics_row["timeStamp"])
    return {
        "lidar_frame": lidar_frame,
        "metrics_index": metrics_index,
        "metrics_row": metrics_row,
        "metrics_timestamp": metrics_timestamp,
        "delta_seconds": lidar_frame.timestamp - metrics_timestamp,
    }


def match_lidar_frames_to_metrics(
    sequence: "LakesideSequence", limit: int | None = None
) -> list[dict[str, Any]]:
    """Match lidar frames to nearest metrics rows by Unix timestamp."""

    metrics = sequence.load_metrics()
    metric_timestamps = metrics.timestamps().tolist()
    lidar_frames = sequence.iter_lidar()
    if limit is not None:
        lidar_frames = islice(lidar_frames, max(limit, 0))

    matches: list[dict[str, Any]] = []
    for lidar_frame in lidar_frames:
        metrics_index = nearest_timestamp(metric_timestamps, lidar_frame.timestamp)
        if metrics_index is None:
            continue
        metrics_row = metrics.dataframe.iloc[metrics_index]
        metrics_timestamp = float(metrics_row["timeStamp"])
        matches.append(
            {
                "lidar_frame": lidar_frame,
                "metrics_index": metrics_index,
                "metrics_row": metrics_row,
                "metrics_timestamp": metrics_timestamp,
                "delta_seconds": lidar_frame.timestamp - metrics_timestamp,
            }
        )
    return matches
