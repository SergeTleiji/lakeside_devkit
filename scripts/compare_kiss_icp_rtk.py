#!/usr/bin/env python3
"""Compare KISS-ICP and RTK traveled distance over consecutive RTK 100 m sections.

Install: python3 -m pip install numpy pyproj
Example:
    python3 compare_kiss_icp_rtk.py --icp poses.csv --pcd-dir scans --rtk rtk.csv \
        --rtk-time-column timestamp --output-dir comparison_results

Only translations are used. Distances are sums of successive translation-step
lengths, NOT the distance between section endpoints. No spatial registration or
orientation comparison is performed. Both files must describe the same run and
use synchronized clocks. RTK rows with latRTK=longRTK=0 are automatically
filtered as no-fix samples before timestamp parsing or distance accumulation.
The RTK CSV must also contain RTK_Status; statuses 2 and 3 are retained by
default. Add --skip-rtk-status-2 to retain only full RTK fixes (status 3).
Rejected fixes split the recording into valid time intervals. Only traveled
distance inside those intervals is accumulated. Sections continue accumulating
across exclusions without counting travel during them. Use --segment-m 100
(default), --segment-m 50, or another length; only the final remainder is partial.
CSV duration_s counts only included time; elapsed_duration_s spans the actual
start/end timestamps, and excluded_duration_s reports exclusions within that span.
Both trajectories are position-resampled at common times at least one second
apart before computing distances and speeds. --sample-interval can increase
that minimum. Short valid fragments are excluded and reported; a short final
tail of a longer fragment is merged into its last sampling step.
The RTK measurement rate is independent of the CSV logging rate (default:
--rtk-fix-rate 5 Hz). Consecutive held copies of the same RTK coordinates are
coalesced before interpolation. Stationary periods are preserved; the log's
held tail beyond the final measurement is excluded from both trajectories.
See README_kiss_icp_rtk.md for the complete method.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from pathlib import Path
from typing import Iterator

try:
    import numpy as np
    from pyproj import CRS, Transformer
    from pyproj.exceptions import ProjError
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install with: python3 -m pip install numpy pyproj"
    ) from exc


NANOSECONDS = 1_000_000_000
INT64 = np.iinfo(np.int64)
UNIT_FACTORS = {"s": Decimal("1e9"), "ms": Decimal("1e6"),
                "us": Decimal("1e3"), "ns": Decimal(1)}
DISTANCE_TOLERANCE_M = 1e-6  # Floating-point boundary tolerance, not a noise filter.
TIME_NAMES = {
    "t", "time", "timestamp", "unixtime", "unixtimestamp", "timeunix",
    "timestamputc", "utctime", "epochtime", "epoch", "timestampunix",
    "times", "timems", "timeus", "timens", "timestamps", "timestampms",
    "timestampus", "timestampns", "timestampseconds", "timestampsec",
}
SECTION_FIELDS = [
    "section", "rtk_valid_interval", "section_type", "rtk_start_m", "rtk_end_m",
    "start_timestamp_s", "end_timestamp_s", "duration_s", "elapsed_duration_s",
    "excluded_duration_s", "valid_interval_count", "rtk_distance_m",
    "icp_distance_m", "signed_error_m", "absolute_error_m",
    "signed_error_pct", "absolute_error_pct",
]


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def table_rows(path: Path) -> Iterator[tuple[int, list[str]]]:
    """Read a comma, semicolon, tab, or whitespace table; skip blank/# lines."""
    delimiter = None
    decided = False
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if not decided:
                counts = {c: line.count(c) for c in (",", ";", "\t")}
                candidate = max(counts, key=counts.get)
                delimiter = candidate if counts[candidate] else None
                decided = True
            row = (next(csv.reader([line], delimiter=delimiter))
                   if delimiter else line.split())
            yield line_number, [value.strip() for value in row]


def column_map(header: list[str], path: Path) -> dict[str, int]:
    names = [normalize(name) for name in header]
    if len(set(names)) != len(names) or "" in names:
        raise ValueError(f"{path}: empty or duplicate column names in header.")
    return {name: i for i, name in enumerate(names)}


def read_icp_poses(path: Path) -> np.ndarray:
    """Keep every pose row, in its original order; ignore t and quaternions."""
    rows = table_rows(path)
    first = next(rows, None)
    if first is None:
        raise ValueError(f"{path}: ICP file is empty.")
    first_names = [normalize(value) for value in first[1]]
    if all(name in first_names for name in ("x", "y", "z")):
        columns = column_map(first[1], path)
        xyz_columns = [columns[name] for name in ("x", "y", "z")]
        data_rows = rows
    else:
        xyz_columns = [1, 2, 3]
        data_rows = itertools.chain([first], rows)
    if len(first[1]) != 8:
        raise ValueError(f"{path}: expected 8 columns: t x y z qx qy qz qw.")
    positions = []
    for line_number, row in data_rows:
        if len(row) != 8:
            raise ValueError(f"{path}:{line_number}: expected 8 columns, got {len(row)}.")
        try:
            xyz = [float(row[i]) for i in xyz_columns]
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid ICP translation.") from exc
        if not all(math.isfinite(value) for value in xyz):
            raise ValueError(f"{path}:{line_number}: non-finite ICP translation.")
        positions.append(xyz)
    if len(positions) < 2:
        raise ValueError("At least two ICP poses are required.")
    return np.asarray(positions, dtype=np.float64)


def checked_ns(value: Decimal, context: str) -> int:
    if not value.is_finite():
        raise ValueError(f"{context}: timestamp is not finite.")
    result = int(value.to_integral_value(rounding=ROUND_HALF_EVEN))
    if not INT64.min <= result <= INT64.max:
        raise ValueError(f"{context}: timestamp is outside the supported nanosecond range.")
    return result


def format_timestamp(ns: int) -> str:
    sign = "-" if ns < 0 else ""
    whole, fraction = divmod(abs(int(ns)), NANOSECONDS)
    return f"{sign}{whole}.{fraction:09d}"


def read_pcd_timestamps(directory: Path, recursive: bool = False
                        ) -> tuple[np.ndarray, list[Path]]:
    if not directory.is_dir():
        raise ValueError(f"PCD directory does not exist: {directory}")
    candidates = directory.rglob("*") if recursive else directory.iterdir()
    entries = []
    for path in candidates:
        if not path.is_file() or path.suffix.lower() != ".pcd":
            continue
        if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", path.stem):
            raise ValueError(f"PCD filename must be a timestamp in seconds: {path.name}")
        ns = checked_ns(Decimal(path.stem) * UNIT_FACTORS["s"], str(path))
        entries.append((ns, path))
    if not entries:
        raise ValueError(f"No PCD files found in {directory}.")
    entries.sort(key=lambda item: item[0])  # Numeric, not lexicographic ordering.
    for previous, current in zip(entries, entries[1:]):
        if previous[0] == current[0]:
            raise ValueError(
                f"Duplicate PCD timestamp: {previous[1]} and {current[1]}."
            )
    return (np.array([item[0] for item in entries], dtype=np.int64),
            [item[1] for item in entries])


def parse_iso_ns(value: str) -> int:
    """Preserve fractional nanoseconds; a missing timezone means UTC."""
    match = re.fullmatch(
        r"(\d{4}-\d{2}-\d{2}[Tt ]\d{2}:\d{2}:\d{2})"
        r"(?:\.(\d+))?(Z|z|[+-]\d{2}:?\d{2})?", value.strip()
    )
    if match is None:
        raise ValueError(f"Invalid ISO timestamp: {value!r}")
    whole, fraction, zone = match.groups()
    zone = "+00:00" if zone is None or zone.upper() == "Z" else zone
    instant = datetime.fromisoformat(whole.replace("t", "T") + zone)
    delta = instant.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    ns = Decimal((delta.days * 86400 + delta.seconds) * NANOSECONDS)
    if fraction:
        ns += Decimal("0." + fraction) * NANOSECONDS
    return checked_ns(ns, value)


def parse_rtk_times(values: list[str], unit: str, offset: str
                    ) -> tuple[np.ndarray, str]:
    try:
        numeric = [Decimal(value) for value in values]
    except InvalidOperation:
        numeric = None
    if unit == "auto":
        if numeric is None:
            unit = "iso"
        else:
            if not all(value.is_finite() for value in numeric):
                raise ValueError("RTK timestamps contain NaN or infinity.")
            magnitude = float(np.median([float(abs(value)) for value in numeric]))
            unit = ("ns" if magnitude >= 1e17 else "us" if magnitude >= 1e14
                    else "ms" if magnitude >= 1e11 else "s")
    if unit == "iso":
        timestamps = [parse_iso_ns(value) for value in values]
    else:
        if numeric is None:
            raise ValueError(f"RTK time unit {unit!r} requires numeric timestamps.")
        timestamps = [checked_ns(value * UNIT_FACTORS[unit], "RTK")
                      for value in numeric]
    offset_ns = checked_ns(Decimal(offset) * NANOSECONDS, "RTK time offset")
    shifted = [checked_ns(Decimal(value + offset_ns), "Shifted RTK")
               for value in timestamps]
    return np.array(shifted, dtype=np.int64), unit


def quality_breaks(times_ns: np.ndarray, source_lines: np.ndarray,
                   rejected_rows: list[tuple[int, str]], unit: str,
                   time_offset: str) -> np.ndarray:
    """Mark accepted-fix pairs separated by a rejected fix; never bridge them.

    Use rejected-fix timestamps, so this also supports time-reordered CSVs.
    If a rejected timestamp is unusable, conservatively use its CSV position.
    Leading/trailing unset rows do not split an otherwise valid recording.
    """
    breaks = np.zeros(len(times_ns), dtype=bool)
    rejected_ns, fallback_lines = [], []
    offset_ns = checked_ns(Decimal(time_offset) * NANOSECONDS, "RTK time offset")
    first_line, last_line = int(source_lines.min()), int(source_lines.max())
    for line, raw_time in rejected_rows:
        try:
            value = (parse_iso_ns(raw_time) if unit == "iso" else
                     checked_ns(Decimal(raw_time) * UNIT_FACTORS[unit], "Rejected RTK"))
            value = checked_ns(Decimal(value + offset_ns), "Rejected RTK")
        except (ValueError, InvalidOperation, OverflowError):
            fallback_lines.append(line)
            continue
        if int(times_ns[0]) < value < int(times_ns[-1]):
            rejected_ns.append(value)
        elif value == offset_ns and first_line < line < last_line:
            # For example, an internal no-fix row whose timestamp is also zero.
            fallback_lines.append(line)
    if rejected_ns:
        rejected = np.asarray(rejected_ns, dtype=np.int64)
        right = np.searchsorted(times_ns, rejected, side="left")
        breaks[right] = True
        # Conflicting good/bad fixes at one internal timestamp exclude both sides.
        exact = right[times_ns[right] == rejected]
        breaks[exact + 1] = True
    if fallback_lines:
        lines = np.sort(np.asarray(fallback_lines, dtype=int))
        low = np.minimum(source_lines[:-1], source_lines[1:])
        high = np.maximum(source_lines[:-1], source_lines[1:])
        breaks[1:] |= (np.searchsorted(lines, high, side="left")
                       > np.searchsorted(lines, low, side="right"))
    return breaks


def read_rtk(path: Path, time_column: str | None, time_unit: str,
             time_offset: str, skip_status_2: bool = False
             ) -> tuple[np.ndarray, np.ndarray, dict]:
    rows = table_rows(path)
    first = next(rows, None)
    if first is None:
        raise ValueError(f"{path}: RTK file is empty.")
    header = first[1]
    columns = column_map(header, path)
    required = ("latrtk", "longrtk", "altrtk")
    if not all(name in columns for name in required):
        raise ValueError(f"{path}: RTK header must contain latRTK, longRTK, altRTK.")
    if "rtkstatus" not in columns:
        raise ValueError(f"{path}: RTK header must contain RTK_Status.")
    if time_column:
        key = normalize(time_column)
        if key not in columns:
            raise ValueError(f"RTK time column {time_column!r} not found. Columns: {header}")
    else:
        matches = [name for name in columns if name in TIME_NAMES]
        if len(matches) != 1:
            raise ValueError(
                "Cannot unambiguously detect the RTK timestamp column. "
                f"Use --rtk-time-column COLUMN. Available columns: {header}"
            )
        key = matches[0]
    times, coordinates, source_lines, zero_fix_lines = [], [], [], []
    status_filtered_lines = []
    rejected_rows = []
    allowed_statuses = (3,) if skip_status_2 else (2, 3)
    allowed_status_text = "/".join(str(value) for value in allowed_statuses)
    input_rows = 0
    for line_number, row in rows:
        input_rows += 1
        if len(row) != len(header):
            raise ValueError(f"{path}:{line_number}: inconsistent number of columns.")
        # Use differential float (2) or full RTK (3), or only 3 if requested. Filter
        # before parsing coordinates/timestamps, which may be unset for
        # rejected rows. Decimal accepts 2.0/3.0 without rounding 2.5 to 2.
        try:
            status = Decimal(row[columns["rtkstatus"]])
        except InvalidOperation:
            status = None
        if status is None or not status.is_finite() or status not in allowed_statuses:
            status_filtered_lines.append(line_number)
            rejected_rows.append((line_number, row[columns[key]]))
            continue
        try:
            lat, lon = [float(row[columns[name]]) for name in required[:2]]
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid RTK coordinates.") from exc
        # This log uses lat=lon=0 to mean "no RTK fix". Remove these rows
        # BEFORE parsing their timestamps (which can also be zero/duplicated),
        # selecting the UTM zone, projecting, or accumulating path length.
        # Zero latitude OR longitude alone remains a valid geographic value.
        if lat == 0.0 and lon == 0.0:
            zero_fix_lines.append(line_number)
            rejected_rows.append((line_number, row[columns[key]]))
            continue
        try:
            alt = float(row[columns["altrtk"]])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid RTK altitude.") from exc
        if not all(math.isfinite(value) for value in (lat, lon, alt)):
            raise ValueError(f"{path}:{line_number}: non-finite RTK coordinates.")
        if not (-80 <= lat <= 84 and -180 <= lon <= 180):
            raise ValueError(
                f"{path}:{line_number}: coordinates outside UTM coverage; "
                "latRTK/longRTK must be decimal degrees, altRTK meters."
            )
        times.append(row[columns[key]])
        coordinates.append((lat, lon, alt))
        source_lines.append(line_number)
    if len(times) < 2:
        raise ValueError(
            "At least two valid RTK samples are required after filtering "
            f"{len(status_filtered_lines)} row(s) with RTK_Status outside {allowed_status_text} "
            f"and {len(zero_fix_lines)} zero-coordinate no-fix row(s)."
        )
    timestamps, unit = parse_rtk_times(times, time_unit, time_offset)
    order = np.argsort(timestamps, kind="stable")
    timestamps = timestamps[order]
    if np.any(timestamps[1:] == timestamps[:-1]):
        raise ValueError("RTK has duplicate timestamps; resolve duplicates before comparison.")
    sorted_source_lines = np.asarray(source_lines, dtype=int)[order]
    breaks = quality_breaks(timestamps, sorted_source_lines, rejected_rows, unit, time_offset)
    return timestamps, np.asarray(coordinates, dtype=np.float64)[order], {
        "time_column": header[columns[key]], "time_unit": unit,
        "time_offset_added_s": time_offset,
        "rows_reordered_by_timestamp": bool(np.any(order != np.arange(len(order)))),
        "_source_csv_lines": sorted_source_lines.tolist(),
        "_quality_break_before": breaks.tolist(),
        "_fix_filtering": {
            "input_rows": input_rows, "retained_rows": len(times),
            "zero_fix_rows_removed": len(zero_fix_lines),
            "removed_csv_lines": zero_fix_lines,
            "allowed_rtk_status_values": list(allowed_statuses),
            "status_rows_removed": len(status_filtered_lines),
            "status_removed_csv_lines": status_filtered_lines,
            "status_rule": f"Only RTK_Status {allowed_status_text} is eligible for comparison",
            "rule": "latRTK == 0 AND longRTK == 0 means no RTK fix",
        },
    }


def project_rtk(lat_lon_alt: np.ndarray, epsg: int | None
                ) -> tuple[np.ndarray, dict]:
    """Use ONE WGS84 UTM zone for the entire track; keep altitude in meters."""
    lat, lon, altitude = lat_lon_alt.T
    if epsg is None:
        center_lat, center_lon = float(np.median(lat)), float(np.median(lon))
        zone = min(60, max(1, math.floor((center_lon + 180) / 6) + 1))
        if 56 <= center_lat < 64 and 3 <= center_lon < 12:
            zone = 32  # Norway exception.
        if 72 <= center_lat <= 84 and 0 <= center_lon < 42:
            zone = (31 if center_lon < 9 else 33 if center_lon < 21
                    else 35 if center_lon < 33 else 37)
        epsg = (32600 if center_lat >= 0 else 32700) + zone
    if not (32601 <= epsg <= 32660 or 32701 <= epsg <= 32760):
        raise ValueError("--utm-epsg must be a WGS84 UTM code: 32601..32660 or 32701..32760.")
    crs = CRS.from_epsg(epsg)
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    easting, northing = transformer.transform(lon, lat, errcheck=True)
    points = np.column_stack((easting, northing, altitude))
    if not np.all(np.isfinite(points)):
        raise ValueError("UTM conversion produced invalid coordinates.")
    return points, {"epsg": epsg, "name": crs.name,
                    "altitude": "Input altRTK in meters; no vertical datum conversion."}


def sample_window_times(start_ns: int, end_ns: int, interval_ns: int) -> np.ndarray:
    """Use steps >= the minimum, merging a short tail into the final step."""
    count = (end_ns - start_ns) // interval_ns
    if count < 1:
        return np.empty(0, dtype=np.int64)
    times = start_ns + np.arange(count + 1, dtype=np.int64) * interval_ns
    times[-1] = end_ns
    return times


def rtk_fix_updates(times_ns: np.ndarray, xyz: np.ndarray, source_rows: np.ndarray,
                    fix_period_ns: int
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Recover fix-update knots from sample-and-hold CSV coordinate repeats.

    A changed XYZ uses its first logged timestamp. For a long unchanged run,
    preserve the last stationary fix before the next change, using the nominal
    RTK measurement period. This avoids smearing motion over a long standstill.
    The final held log copies do not extend the last measurement's time support.
    """
    starts = np.r_[0, np.flatnonzero(np.any(xyz[1:] != xyz[:-1], axis=1)) + 1]
    ends = np.r_[starts[1:], len(times_ns)]
    updates, points, before, after = [], [], [], []
    for start, stop in zip(starts, ends):
        first_time = int(times_ns[start])
        updates.append(first_time)
        points.append(xyz[start])
        before.append(int(source_rows[start]))
        after.append(int(source_rows[start]))
        if stop - start < 2:
            continue
        last_logged = int(times_ns[stop - 1])
        if stop < len(times_ns):
            stationary_end = min(last_logged, int(times_ns[stop]) - fix_period_ns)
        else:
            stationary_end = first_time + ((last_logged - first_time) // fix_period_ns) * fix_period_ns
        if stationary_end > first_time:
            upper = int(np.searchsorted(times_ns, stationary_end, side="left"))
            lower = upper if int(times_ns[upper]) == stationary_end else upper - 1
            updates.append(stationary_end)
            points.append(xyz[start])
            before.append(int(source_rows[lower]))
            after.append(int(source_rows[upper]))
    return (np.asarray(updates, dtype=np.int64), np.asarray(points, dtype=float),
            np.asarray(before, dtype=int), np.asarray(after, dtype=int), {
                "consecutive_repeated_coordinate_rows": len(times_ns) - len(starts),
                "redundant_logged_rows_removed": len(times_ns) - len(updates),
                "fix_interpolation_points": len(updates),
            })


def interpolate_positions(times_ns: np.ndarray, xyz: np.ndarray,
                           sample_ns: np.ndarray, source_rows: np.ndarray,
                           source_rows_after: np.ndarray | None = None
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate XYZ, retaining both source-row brackets for each new point."""
    upper = np.searchsorted(times_ns, sample_ns, side="left")
    upper = np.minimum(upper, len(times_ns) - 1)
    exact = times_ns[upper] == sample_ns
    lower = np.where(exact, upper, np.maximum(upper - 1, 0))
    left, right = int(lower.min()), int(upper.max()) + 1
    origin = int(sample_ns[0])
    raw_times = (times_ns[left:right] - origin).astype(np.float64) / NANOSECONDS
    target_times = (sample_ns - origin).astype(np.float64) / NANOSECONDS
    points = np.column_stack([
        np.interp(target_times, raw_times, xyz[left:right, axis]) for axis in range(3)])
    upper_rows = source_rows if source_rows_after is None else source_rows_after
    return points, source_rows[lower], upper_rows[upper]


def inspect_trajectory_steps(name: str, times_ns: np.ndarray, xyz: np.ndarray,
                             interval_ids: np.ndarray, source_before: np.ndarray,
                             source_after: np.ndarray, source_name: str,
                             dimensions: int, output_dir: Path,
                             speed_limit: float = 0.0,
                             geographic_points: np.ndarray | None = None) -> dict:
    """Write every resampled step and its speed, never crossing quality breaks."""
    indices = np.flatnonzero(interval_ids[:-1] == interval_ids[1:])
    delta = np.diff(xyz, axis=0)
    dt = np.diff(times_ns).astype(np.float64) / NANOSECONDS
    distance = np.linalg.norm(delta[:, :dimensions], axis=1)
    speed = distance / dt
    bad = indices[speed[indices] > speed_limit] if speed_limit > 0 else np.array([], dtype=int)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_path = output_dir / f"{name.lower()}_step_diagnostics.csv"
    fields = ["rtk_valid_interval", f"from_{source_name}", f"from_{source_name}_after",
              f"to_{source_name}", f"to_{source_name}_after",
              "from_timestamp_s", "to_timestamp_s", "dt_s",
              "from_x_m", "from_y_m", "from_z_m", "to_x_m", "to_y_m", "to_z_m",
              "horizontal_step_m", "vertical_change_m", "selected_step_m",
              "horizontal_speed_m_s", "selected_speed_m_s", "selected_speed_km_h",
              "exceeds_speed_limit"]
    if geographic_points is not None:
        fields += ["from_latRTK", "from_longRTK", "from_altRTK",
                   "to_latRTK", "to_longRTK", "to_altRTK"]
    with diagnostic_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in indices:
            horizontal = float(np.linalg.norm(delta[index, :2]))
            row = {
                "rtk_valid_interval": int(interval_ids[index]),
                f"from_{source_name}": int(source_before[index]),
                f"from_{source_name}_after": int(source_after[index]),
                f"to_{source_name}": int(source_before[index + 1]),
                f"to_{source_name}_after": int(source_after[index + 1]),
                "from_timestamp_s": format_timestamp(times_ns[index]),
                "to_timestamp_s": format_timestamp(times_ns[index + 1]),
                "dt_s": float(dt[index]),
                **dict(zip(("from_x_m", "from_y_m", "from_z_m"), xyz[index])),
                **dict(zip(("to_x_m", "to_y_m", "to_z_m"), xyz[index + 1])),
                "horizontal_step_m": horizontal, "vertical_change_m": float(delta[index, 2]),
                "selected_step_m": float(distance[index]),
                "horizontal_speed_m_s": horizontal / dt[index],
                "selected_speed_m_s": float(speed[index]),
                "selected_speed_km_h": float(speed[index] * 3.6),
                "exceeds_speed_limit": bool(speed_limit > 0 and speed[index] > speed_limit),
            }
            if geographic_points is not None:
                row.update(dict(zip(("from_latRTK", "from_longRTK", "from_altRTK"),
                                    geographic_points[index])))
                row.update(dict(zip(("to_latRTK", "to_longRTK", "to_altRTK"),
                                    geographic_points[index + 1])))
            writer.writerow(row)
    total_time = float(np.sum(dt[indices]))
    worst = indices[np.argsort(speed[indices], kind="stable")[-3:]][::-1]
    return {
        "maximum_step_m": float(np.max(distance[indices], initial=0.0)),
        "maximum_speed_m_s": float(np.max(speed[indices], initial=0.0)),
        "mean_speed_m_s": float(np.sum(distance[indices]) / total_time) if total_time else 0.0,
        "median_speed_m_s": float(np.median(speed[indices])) if len(indices) else 0.0,
        "minimum_step_duration_s": float(np.min(dt[indices])) if len(indices) else None,
        "maximum_step_duration_s": float(np.max(dt[indices], initial=0.0)),
        "total_sampled_duration_s": total_time,
        "intervals_inspected": len(indices),
        "maximum_allowed_speed_m_s": speed_limit if speed_limit else None,
        "speed_limit_exceeded_intervals": len(bad),
        "worst_speed_intervals": [{
            "from_timestamp_s": format_timestamp(times_ns[index]),
            "to_timestamp_s": format_timestamp(times_ns[index + 1]),
            "step_m": float(distance[index]), "dt_s": float(dt[index]),
            "speed_m_s": float(speed[index]),
        } for index in worst],
        "diagnostic_file": str(diagnostic_path),
    }


@dataclass
class DistanceTrack:
    times_s: np.ndarray
    cumulative_m: np.ndarray


@dataclass(frozen=True)
class ComparisonWindow:
    interval: int
    start_ns: int
    end_ns: int
    active_start_ns: int
    active_end_ns: int


def distance_track(times_ns: np.ndarray, positions: np.ndarray,
                   origin_ns: int, dimensions: int) -> DistanceTrack:
    # Subtract integer epoch origin BEFORE conversion to floating point.
    times = (times_ns - origin_ns).astype(np.float64) / NANOSECONDS
    steps = np.linalg.norm(np.diff(positions[:, :dimensions], axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(steps)]
    if not np.all(np.isfinite(cumulative)):
        raise ValueError("Trajectory distance overflowed; check position units and values.")
    return DistanceTrack(times, cumulative)


def check_time_gaps(track: DistanceTrack | np.ndarray, start: float, end: float,
                    limit: float, name: str) -> float:
    times = track.times_s if isinstance(track, DistanceTrack) else track
    touching = (times[:-1] < end) & (times[1:] > start)
    gaps = np.diff(times)[touching]
    maximum = float(np.max(gaps, initial=0.0))
    if limit > 0 and maximum > limit + 1e-9:
        raise ValueError(
            f"{name}: a {maximum:.6f} s sampling gap touches the comparison interval "
            f"(limit {limit:g} s). Inspect the data or set --max-time-gap explicitly; "
            "0 disables this check. Gaps are never silently skipped."
        )
    return maximum


def crop_distance(track: DistanceTrack, start: float, end: float) -> DistanceTrack:
    interior = (track.times_s > start) & (track.times_s < end)
    times = np.r_[start, track.times_s[interior], end]
    distance = np.interp(times, track.times_s, track.cumulative_m)
    return DistanceTrack(times, distance - distance[0])


def first_time_at_distance(track: DistanceTrack, target_m: float) -> float:
    """Invert cumulative RTK distance; plateaus use the first arrival time."""
    index = int(np.searchsorted(track.cumulative_m, target_m, side="left"))
    if index == 0:
        return float(track.times_s[0])
    if index >= len(track.times_s):
        raise ValueError("Internal error: requested distance exceeds RTK coverage.")
    d0, d1 = track.cumulative_m[index - 1:index + 1]
    fraction = (target_m - d0) / (d1 - d0)
    return float(track.times_s[index - 1] + fraction * (
        track.times_s[index] - track.times_s[index - 1]))


def compare_sections(icp: DistanceTrack, rtk: DistanceTrack, start: float,
                     end: float, origin_ns: int, section_m: float,
                     include_partial: bool = True,
                     time_windows: list[ComparisonWindow] | None = None) -> list[dict]:
    reference = crop_distance(rtk, start, end)
    total = float(reference.cumulative_m[-1])
    if total <= DISTANCE_TOLERANCE_M:
        raise ValueError("RTK traveled distance in the overlapping interval is zero.")
    full_count = int(math.floor(total / section_m))
    if abs(total - (full_count + 1) * section_m) <= DISTANCE_TOLERANCE_M:
        full_count += 1
    boundaries = [(0.0, start)]
    for i in range(1, full_count + 1):
        target = min(i * section_m, total)
        time = (end if abs(total - target) <= DISTANCE_TOLERANCE_M
                else first_time_at_distance(reference, target))
        boundaries.append((target, time))
    if include_partial and total - boundaries[-1][0] > DISTANCE_TOLERANCE_M:
        boundaries.append((total, end))
    if time_windows:
        active_starts_ns = np.array([window.active_start_ns for window in time_windows], dtype=np.int64)
        active_ends_ns = np.array([window.active_end_ns for window in time_windows], dtype=np.int64)
    result = []
    for index, ((d0, t0), (d1, t1)) in enumerate(zip(boundaries, boundaries[1:]), 1):
        rtk_distance = d1 - d0
        icp_distance = float(np.interp(t1, icp.times_s, icp.cumulative_m)
                             - np.interp(t0, icp.times_s, icp.cumulative_m))
        error = icp_distance - rtk_distance
        real_start_ns = origin_ns + round(t0 * NANOSECONDS)
        real_end_ns = origin_ns + round(t1 * NANOSECONDS)
        interval_label, interval_count = 1, 1
        if time_windows:
            # Boundaries use active time, with rejected periods removed. At a
            # join, a section ends on the earlier window and the next one starts
            # on the later window. No travel through the real-time gap is added.
            active_start_ns, active_end_ns = round(t0 * NANOSECONDS), round(t1 * NANOSECONDS)
            first = min(int(np.searchsorted(active_ends_ns, active_start_ns, side="right")),
                        len(time_windows) - 1)
            last = max(first, int(np.searchsorted(active_starts_ns, active_end_ns, side="left")) - 1)
            first_window, last_window = time_windows[first], time_windows[last]
            real_start_ns = min(first_window.end_ns, max(first_window.start_ns,
                first_window.start_ns + active_start_ns - first_window.active_start_ns))
            real_end_ns = min(last_window.end_ns, max(last_window.start_ns,
                last_window.start_ns + active_end_ns - last_window.active_start_ns))
            interval_count = last - first + 1
            interval_label = (first_window.interval if first == last else
                              f"{first_window.interval}-{last_window.interval}")
        elapsed_duration = (real_end_ns - real_start_ns) / NANOSECONDS
        result.append({
            "section": index,
            "rtk_valid_interval": interval_label,
            "section_type": "full" if index <= full_count else "partial",
            "rtk_start_m": d0, "rtk_end_m": d1,
            "start_timestamp_s": format_timestamp(real_start_ns),
            "end_timestamp_s": format_timestamp(real_end_ns),
            "duration_s": t1 - t0, "elapsed_duration_s": elapsed_duration,
            "excluded_duration_s": max(0.0, elapsed_duration - (t1 - t0)),
            "valid_interval_count": interval_count, "rtk_distance_m": rtk_distance,
            "icp_distance_m": icp_distance, "signed_error_m": error,
            "absolute_error_m": abs(error),
            "signed_error_pct": 100 * error / rtk_distance,
            "absolute_error_pct": 100 * abs(error) / rtk_distance,
        })
    if not result:
        raise ValueError(
            f"RTK travels only {total:.6f} m; no full {section_m:g} m section. "
            "Omit --drop-partial or choose a smaller --segment-m."
        )
    return result


def mean_metrics(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    names = ["rtk_distance_m", "icp_distance_m", "signed_error_m",
             "absolute_error_m", "signed_error_pct", "absolute_error_pct"]
    return {"section_count": len(rows),
            **{f"mean_{name}": float(np.mean([row[name] for row in rows]))
               for name in names}}


def print_table(rows: list[dict], section_m: float) -> None:
    if not rows:
        print("\nNo eligible distance sections; no comparison averages were computed.")
        return
    print("\nSection  Intervals  Type       RTK range [m]      RTK [m]    ICP [m]  Error [m]  Abs [m]  Error [%]")
    for row in rows:
        print(f"{row['section']:>7}  {row.get('rtk_valid_interval', '-'):>8}  {row['section_type']:<7}  "
              f"{row['rtk_start_m']:>8.2f}-{row['rtk_end_m']:<8.2f}  "
              f"{row['rtk_distance_m']:>9.3f}  {row['icp_distance_m']:>9.3f}  "
              f"{row['signed_error_m']:>9.3f}  {row['absolute_error_m']:>7.3f}  "
              f"{row['signed_error_pct']:>9.3f}")
    full = mean_metrics([row for row in rows if row["section_type"] == "full"])
    if full:
        print(f"\nAverage over {full['section_count']} full {section_m:g} m sections:")
        print(f"  RTK distance: {full['mean_rtk_distance_m']:.6f} m; "
              f"ICP distance: {full['mean_icp_distance_m']:.6f} m")
        print(f"  Mean signed error: {full['mean_signed_error_m']:.6f} m "
              f"({full['mean_signed_error_pct']:.6f}%)")
        print(f"  Mean absolute error: {full['mean_absolute_error_m']:.6f} m "
              f"({full['mean_absolute_error_pct']:.6f}%)")
    else:
        partial = mean_metrics(rows)
        print(f"\nNo full {section_m:g} m section: only the final partial section is available.")
        print(f"  Mean partial-section absolute error: "
              f"{partial['mean_absolute_error_m']:.6f} m "
              f"({partial['mean_absolute_error_pct']:.6f}%)")


def run(args: argparse.Namespace) -> None:
    for value, label in ((args.segment_m, "--segment-m"),
                         (args.sample_interval, "--sample-interval"),
                         (args.rtk_fix_rate, "--rtk-fix-rate"),
                         (args.max_time_gap, "--max-time-gap"),
                         (args.max_rtk_speed, "--max-rtk-speed")):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{label} must be finite and nonnegative.")
    if args.segment_m <= DISTANCE_TOLERANCE_M:
        raise ValueError("--segment-m must be greater than 0.000001 m.")
    if args.sample_interval < 1.0:
        raise ValueError("--sample-interval must be at least 1 second.")
    sample_interval_ns = checked_ns(Decimal(str(args.sample_interval)) * NANOSECONDS,
                                    "Sampling interval")
    if args.rtk_fix_rate <= 0:
        raise ValueError("--rtk-fix-rate must be positive.")
    fix_period_ns = checked_ns(Decimal(NANOSECONDS) / Decimal(str(args.rtk_fix_rate)),
                              "RTK measurement period")
    if fix_period_ns < 1:
        raise ValueError("--rtk-fix-rate exceeds nanosecond timestamp resolution.")
    icp_xyz = read_icp_poses(args.icp)
    icp_ns, pcd_paths = read_pcd_timestamps(args.pcd_dir, args.recursive)
    if len(icp_xyz) != len(icp_ns):
        raise ValueError(
            f"ICP/PCD count mismatch: {len(icp_xyz)} pose rows versus "
            f"{len(icp_ns)} PCD files. Every row must correspond to exactly one "
            "chronologically sorted PCD. No rows/files were discarded."
        )
    rtk_ns, rtk_lla, time_metadata = read_rtk(
        args.rtk, args.rtk_time_column, args.rtk_time_unit, args.rtk_time_offset,
        args.skip_rtk_status_2)
    fix_filtering = time_metadata.pop("_fix_filtering")
    rtk_input_count = fix_filtering["input_rows"]
    rtk_source_lines = np.asarray(time_metadata.pop("_source_csv_lines"), dtype=int)
    break_before = np.asarray(time_metadata.pop("_quality_break_before"), dtype=bool)
    start_ns = max(int(icp_ns[0]), int(rtk_ns[0]))
    end_ns = min(int(icp_ns[-1]), int(rtk_ns[-1]))
    if start_ns >= end_ns:
        raise ValueError(
            "ICP and RTK have no overlapping time interval. "
            f"ICP: {format_timestamp(icp_ns[0])}..{format_timestamp(icp_ns[-1])}; "
            f"RTK: {format_timestamp(rtk_ns[0])}..{format_timestamp(rtk_ns[-1])}. "
            "Check RTK column, units, clock convention, and --rtk-time-offset."
        )
    # Keep the origin inside the overlap: also limits floating-point time magnitudes.
    origin_ns = start_ns
    rtk_xyz, utm_metadata = project_rtk(rtk_lla, args.utm_epsg)
    dimensions = 3 if args.dimensions == "3d" else 2
    raw_icp_times = (icp_ns - origin_ns).astype(np.float64) / NANOSECONDS
    raw_rtk_times = (rtk_ns - origin_ns).astype(np.float64) / NANOSECONDS
    icp_source_rows = np.arange(1, len(icp_ns) + 1, dtype=int)
    start, end = 0.0, (end_ns - start_ns) / NANOSECONDS
    # Integrate each accepted-fix run independently, then concatenate the scalar
    # distance increments on an active-time axis. Do NOT concatenate positions:
    # that would introduce an artificial connecting displacement across outages.
    boundaries = [0, *np.flatnonzero(break_before).tolist(), len(rtk_ns)]
    gaps = {"ICP": 0.0, "RTK": 0.0}
    sections, valid_intervals, excluded_intervals, short_intervals = [], [], [], []
    rtk_parts, icp_parts, time_windows = [], [], []
    diagnostic_parts = {key: [] for key in (
        "time", "rtk_xyz", "icp_xyz", "interval_id",
        "rtk_before", "rtk_after", "icp_before", "icp_after")}
    rtk_total = eligible_icp_total = 0.0
    valid_duration_ns = 0
    repeat_counts = {"consecutive_repeated_coordinate_rows": 0,
                     "redundant_logged_rows_removed": 0, "fix_interpolation_points": 0}
    held_tail_excluded_ns = 0
    for right in np.flatnonzero(break_before):
        excluded_start = max(start_ns, int(rtk_ns[right - 1]))
        excluded_end = min(end_ns, int(rtk_ns[right]))
        if excluded_start < excluded_end:
            excluded_intervals.append({
                "start_timestamp_s": format_timestamp(excluded_start),
                "end_timestamp_s": format_timestamp(excluded_end),
                "duration_s": (excluded_end - excluded_start) / NANOSECONDS,
            })
    for left, right in zip(boundaries, boundaries[1:]):
        if right - left < 2:
            continue
        raw_window_start_ns = max(start_ns, int(rtk_ns[left]))
        raw_window_end_ns = min(end_ns, int(rtk_ns[right - 1]))
        if raw_window_start_ns >= raw_window_end_ns:
            continue
        fix_ns, fix_xyz, fix_before, fix_after, repeats = rtk_fix_updates(
            rtk_ns[left:right], rtk_xyz[left:right], rtk_source_lines[left:right], fix_period_ns)
        for key in repeat_counts:
            repeat_counts[key] += repeats[key]
        window_start_ns = max(raw_window_start_ns, int(fix_ns[0]))
        window_end_ns = max(window_start_ns, min(raw_window_end_ns, int(fix_ns[-1])))
        held_tail = (raw_window_end_ns - raw_window_start_ns) - (window_end_ns - window_start_ns)
        held_tail_excluded_ns += held_tail
        sample_ns = sample_window_times(window_start_ns, window_end_ns, sample_interval_ns)
        if not len(sample_ns):
            short_intervals.append({
                "start_timestamp_s": format_timestamp(window_start_ns),
                "end_timestamp_s": format_timestamp(window_end_ns),
                "duration_s": (window_end_ns - window_start_ns) / NANOSECONDS,
                "raw_logged_duration_s": (raw_window_end_ns - raw_window_start_ns) / NANOSECONDS,
                "held_tail_excluded_s": held_tail / NANOSECONDS,
                "reason": "Shorter than the minimum position-sampling interval",
            })
            continue
        window_start = (window_start_ns - origin_ns) / NANOSECONDS
        window_end = (window_end_ns - origin_ns) / NANOSECONDS
        # Check genuine logging gaps in RAW timestamps, before intentional
        # resampling. A coarser requested sampling interval is not a logging gap.
        for name, raw_times in (("ICP", raw_icp_times), ("RTK", raw_rtk_times[left:right])):
            gaps[name] = max(gaps[name], check_time_gaps(
                raw_times, window_start, window_end, args.max_time_gap, name))
        sampled_rtk, rtk_before, rtk_after = interpolate_positions(
            fix_ns, fix_xyz, sample_ns, fix_before, fix_after)
        sampled_icp, icp_before, icp_after = interpolate_positions(
            icp_ns, icp_xyz, sample_ns, icp_source_rows)
        # Compute new distances from the sampled POSITIONS, not interpolated
        # high-frequency cumulative distances, so sub-second motion is not summed.
        reference = distance_track(sample_ns, sampled_rtk, origin_ns, dimensions)
        icp_reference = distance_track(sample_ns, sampled_icp, origin_ns, dimensions)
        distance = float(reference.cumulative_m[-1])
        icp_distance = float(icp_reference.cumulative_m[-1])
        interval_id = len(valid_intervals) + 1
        for key, values in (
                ("time", sample_ns), ("rtk_xyz", sampled_rtk), ("icp_xyz", sampled_icp),
                ("interval_id", np.full(len(sample_ns), interval_id, dtype=int)),
                ("rtk_before", rtk_before), ("rtk_after", rtk_after),
                ("icp_before", icp_before), ("icp_after", icp_after)):
            diagnostic_parts[key].append(values)
        active_end_ns = valid_duration_ns + window_end_ns - window_start_ns
        time_windows.append(ComparisonWindow(
            interval_id, window_start_ns, window_end_ns, valid_duration_ns, active_end_ns))
        first_vertex = 0 if not rtk_parts else 1  # Remove the duplicate active-time join.
        for parts, reference_track, distance_offset in (
                (rtk_parts, reference, rtk_total), (icp_parts, icp_reference, eligible_icp_total)):
            active_times = reference_track.times_s - window_start + valid_duration_ns / NANOSECONDS
            active_times[0] = valid_duration_ns / NANOSECONDS
            active_times[-1] = active_end_ns / NANOSECONDS
            parts.append(DistanceTrack(active_times[first_vertex:],
                (reference_track.cumulative_m + distance_offset)[first_vertex:]))
        valid_intervals.append({
            "rtk_valid_interval": interval_id,
            "start_timestamp_s": format_timestamp(window_start_ns),
            "end_timestamp_s": format_timestamp(window_end_ns),
            "duration_s": (window_end_ns - window_start_ns) / NANOSECONDS,
            "sampled_points": len(sample_ns),
            "rtk_fix_interpolation_points": len(fix_ns),
            "minimum_sample_interval_s": float(np.min(np.diff(sample_ns))) / NANOSECONDS,
            "maximum_sample_interval_s": float(np.max(np.diff(sample_ns))) / NANOSECONDS,
            "rtk_distance_m": distance, "icp_distance_m": icp_distance,
            "rtk_start_m": rtk_total, "rtk_end_m": rtk_total + distance,
        })
        rtk_total += distance
        eligible_icp_total += icp_distance
        valid_duration_ns = active_end_ns
    valid_duration = valid_duration_ns / NANOSECONDS
    def combined(key: str) -> np.ndarray:
        if diagnostic_parts[key]:
            return np.concatenate(diagnostic_parts[key])
        return np.empty((0, 3), dtype=float) if key.endswith("xyz") else np.empty(0, dtype=np.int64)

    sampled_times = combined("time")
    sampled_rtk_xyz, sampled_icp_xyz = combined("rtk_xyz"), combined("icp_xyz")
    sampled_interval_ids = combined("interval_id")
    inverse = Transformer.from_crs(f"EPSG:{utm_metadata['epsg']}", "EPSG:4326", always_xy=True)
    sample_lon, sample_lat = inverse.transform(sampled_rtk_xyz[:, 0], sampled_rtk_xyz[:, 1], errcheck=True)
    sampled_lla = np.column_stack((sample_lat, sample_lon, sampled_rtk_xyz[:, 2]))
    rtk_checks = inspect_trajectory_steps(
        "RTK", sampled_times, sampled_rtk_xyz, sampled_interval_ids,
        combined("rtk_before"), combined("rtk_after"), "csv_line", dimensions,
        args.output_dir, speed_limit=args.max_rtk_speed, geographic_points=sampled_lla)
    icp_checks = inspect_trajectory_steps(
        "ICP", sampled_times, sampled_icp_xyz, sampled_interval_ids,
        combined("icp_before"), combined("icp_after"), "pose_row", dimensions,
        args.output_dir)
    if rtk_checks["speed_limit_exceeded_intervals"]:
        examples = "; ".join(
            f"{item['from_timestamp_s']}->{item['to_timestamp_s']}: "
            f"{item['step_m']:.3f} m in {item['dt_s']:.6f} s ({item['speed_m_s']:.3f} m/s)"
            for item in rtk_checks["worst_speed_intervals"])
        raise ValueError(
            f"Implausible RTK motion: {rtk_checks['speed_limit_exceeded_intervals']} sampled step(s) "
            f"exceed --max-rtk-speed {args.max_rtk_speed:g} m/s. {examples}. "
            "Both RTK and ICP diagnostics were saved. Inspect their source-row brackets "
            "and coordinate units. No new comparison table was generated."
        )
    if rtk_total > DISTANCE_TOLERANCE_M and (
            not args.drop_partial or rtk_total >= args.segment_m - DISTANCE_TOLERANCE_M):
        accumulated_rtk = DistanceTrack(
            np.concatenate([part.times_s for part in rtk_parts]),
            np.concatenate([part.cumulative_m for part in rtk_parts]))
        accumulated_icp = DistanceTrack(
            np.concatenate([part.times_s for part in icp_parts]),
            np.concatenate([part.cumulative_m for part in icp_parts]))
        sections = compare_sections(
            accumulated_icp, accumulated_rtk, 0.0, valid_duration, origin_ns,
            args.segment_m, not args.drop_partial, time_windows=time_windows)
    total_rtk = sum(row["rtk_distance_m"] for row in sections)
    total_icp = sum(row["icp_distance_m"] for row in sections)
    total_absolute = sum(row["absolute_error_m"] for row in sections)
    raw_rtk_dt = np.diff(rtk_ns)[~break_before[1:]].astype(float) / NANOSECONDS
    log_rate_hz = 1.0 / float(np.median(raw_rtk_dt)) if len(raw_rtk_dt) else None
    summary = {
        "method": "Position-resampled RTK-distance sections accumulated over accepted intervals; exclusions add no distance",
        "dimensions": args.dimensions, "segment_length_m": args.segment_m,
        "inputs": {"icp": str(args.icp.resolve()), "pcd_dir": str(args.pcd_dir.resolve()),
                   "rtk": str(args.rtk.resolve())},
        "counts": {"icp_poses": len(icp_xyz), "pcd_files": len(pcd_paths),
                   "rtk_samples": len(rtk_ns), "rtk_input_samples": rtk_input_count,
                   "sampled_points_per_trajectory": len(sampled_times),
                   "sampled_steps_per_trajectory": rtk_checks["intervals_inspected"],
                   "sections": len(sections), "rtk_valid_intervals": len(valid_intervals)},
        "sampling": {
            "minimum_interval_s": args.sample_interval,
            "policy": "Both XYZ trajectories interpolated at common times; short tails merged into last step",
            "rtk_fix_rate_hz": args.rtk_fix_rate,
            "rtk_logged_rate_hz_estimate": log_rate_hz,
            "rtk_coordinate_hold_processing": repeat_counts,
            "rtk_held_tail_duration_excluded_s": held_tail_excluded_ns / NANOSECONDS,
            "rtk_timestamp_policy": "First logged coordinate update; stationary support inferred from RTK fix rate",
            "short_valid_intervals_skipped": len(short_intervals),
            "short_valid_duration_excluded_s": sum(item["duration_s"] for item in short_intervals),
            "short_interval_details": short_intervals,
            "source_row_brackets": "Interpolated samples list both surrounding CSV lines or ICP pose rows",
        },
        "rtk_fix_filtering": fix_filtering, "rtk_step_checks": rtk_checks,
        "icp_step_checks": icp_checks,
        "rtk_timestamps": time_metadata, "projection": utm_metadata,
        "time_overlap": {"start_timestamp_s": format_timestamp(start_ns),
                         "end_timestamp_s": format_timestamp(end_ns), "duration_s": end},
        "maximum_sampling_gap_s": gaps, "maximum_allowed_gap_s": args.max_time_gap,
        "rtk_quality_exclusions": {
            "intervals": excluded_intervals, "count": len(excluded_intervals),
            "excluded_duration_s": sum(item["duration_s"] for item in excluded_intervals),
            "policy": "No RTK or ICP travel is evaluated across rejected fixes",
        },
        "valid_comparison_intervals": valid_intervals,
        "valid_comparison_duration_s": valid_duration,
        "overlap_rtk_distance_m": rtk_total,
        "eligible_icp_distance_m": eligible_icp_total,
        "excluded_final_remainder_m": max(0.0, rtk_total - total_rtk),
        "full_sections_mean": mean_metrics([r for r in sections if r["section_type"] == "full"]),
        "all_sections_mean": mean_metrics(sections),
        "overall_evaluated": {
            "rtk_distance_m": total_rtk, "icp_distance_m": total_icp,
            "signed_error_m": total_icp - total_rtk,
            "signed_error_pct": 100 * (total_icp - total_rtk) / total_rtk if total_rtk else None,
            "sum_section_absolute_errors_m": total_absolute,
            "distance_weighted_absolute_error_pct": 100 * total_absolute / total_rtk if total_rtk else None,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = args.output_dir / "comparison.csv"
    with comparison_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SECTION_FIELDS)
        writer.writeheader()
        writer.writerows(sections)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with (args.output_dir / "icp_timestamp_mapping.csv").open(
            "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["pose_row_1based", "pcd_file", "timestamp_s", "x", "y", "z"])
        for i, (path, ns, xyz) in enumerate(zip(pcd_paths, icp_ns, icp_xyz), 1):
            writer.writerow([i, str(path.resolve()), format_timestamp(ns), *xyz])
    print(f"Matched {len(icp_xyz)} ICP poses to {len(pcd_paths)} numerically sorted PCD files.")
    print(f"RTK timestamp column: {time_metadata['time_column']} "
          f"(unit: {time_metadata['time_unit']}; added offset: {args.rtk_time_offset} s)")
    print(f"Projection: EPSG:{utm_metadata['epsg']} ({utm_metadata['name']})")
    allowed_status_text = "/".join(str(value) for value in fix_filtering["allowed_rtk_status_values"])
    print(f"RTK status filter (keep RTK_Status {allowed_status_text}): "
          f"removed {fix_filtering['status_rows_removed']} rows.")
    print(f"RTK no-fix rows filtered (latRTK=longRTK=0): "
          f"{fix_filtering['zero_fix_rows_removed']}; retained {len(rtk_ns)} valid samples.")
    print(f"Distance mode: {args.dimensions}; shared time coverage: {end:.6f} s")
    print(f"Position sampling: minimum {args.sample_interval:g} s; "
          f"{rtk_checks['intervals_inspected']} matched RTK/ICP steps.")
    print(f"RTK measurement rate: {args.rtk_fix_rate:g} Hz; "
          f"coalesced {repeat_counts['redundant_logged_rows_removed']} held coordinate copies; "
          f"excluded {held_tail_excluded_ns / NANOSECONDS:.6f} s of held tails.")
    print(f"Skipped {len(short_intervals)} valid fragment(s) shorter than "
          f"{args.sample_interval:g} s "
          f"({sum(item['duration_s'] for item in short_intervals):.6f} s total).")
    print(f"RTK speed: mean {rtk_checks['mean_speed_m_s']:.6f}, "
          f"max {rtk_checks['maximum_speed_m_s']:.6f} m/s; "
          f"ICP speed: mean {icp_checks['mean_speed_m_s']:.6f}, "
          f"max {icp_checks['maximum_speed_m_s']:.6f} m/s.")
    excluded_duration = sum(item["duration_s"] for item in excluded_intervals)
    print(f"Excluded {len(excluded_intervals)} poor-quality RTK interval(s) "
          f"({excluded_duration:.6f} s); comparing {len(valid_intervals)} valid interval(s) "
          f"({valid_duration:.6f} s).")
    print_table(sections, args.segment_m)
    print(f"\nEvaluated totals: RTK {total_rtk:.6f} m; ICP {total_icp:.6f} m")
    if total_rtk:
        print(f"Distance-weighted absolute section error: "
              f"{100 * total_absolute / total_rtk:.6f}%")
    if args.drop_partial and rtk_total - total_rtk > DISTANCE_TOLERANCE_M:
        print(f"Excluded final RTK remainder: {rtk_total - total_rtk:.6f} m")
    print(f"\nSaved: {comparison_path}")
    print(f"Saved: {args.output_dir / 'summary.json'}")
    print(f"Saved: {args.output_dir / 'icp_timestamp_mapping.csv'}")
    print(f"Saved: {args.output_dir / 'rtk_step_diagnostics.csv'}")
    print(f"Saved: {args.output_dir / 'icp_step_diagnostics.csv'}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--icp", type=Path, required=True, help="ICP t x y z qx qy qz qw table.")
    parser.add_argument("--pcd-dir", type=Path, required=True, help="Folder of timestamp.pcd files.")
    parser.add_argument("--rtk", type=Path, required=True, help="RTK table with timestamp, latRTK/longRTK/altRTK, and RTK_Status (keep 2/3).")
    parser.add_argument("--skip-rtk-status-2", action="store_true",
                        help="Exclude float fixes (RTK_Status 2); use only full RTK fixes (3). Default: keep 2 and 3.")
    parser.add_argument("--rtk-time-column", help="RTK timestamp column; default: detect one common name.")
    parser.add_argument("--rtk-time-unit", choices=["auto", "s", "ms", "us", "ns", "iso"],
                        default="auto", help="Numeric epoch units or ISO 8601; default: auto.")
    parser.add_argument("--rtk-time-offset", default="0", metavar="SECONDS",
                        help="Constant seconds ADDED to RTK times; default: 0.")
    parser.add_argument("--segment-m", type=float, default=100.0,
                        help="Accumulated valid RTK distance per section, e.g. 50 or 100 m; default: 100.")
    parser.add_argument("--sample-interval", type=float, default=1.0, metavar="SECONDS",
                        help="Minimum time between sampled positions for BOTH trajectories; must be >= 1 s. Default: 1.")
    parser.add_argument("--rtk-fix-rate", type=float, default=5.0, metavar="HZ",
                        help="RTK measurement rate, independent of CSV logging rate; default: 5 Hz. Used to coalesce held coordinate copies.")
    parser.add_argument("--dimensions", choices=["3d", "2d"], default="3d",
                        help="3d uses xyz/UTM+altitude; 2d uses xy/UTM; default: 3d.")
    parser.add_argument("--utm-epsg", type=int,
                        help="Fixed WGS84 UTM EPSG code; default: select from median RTK location.")
    parser.add_argument("--max-time-gap", type=float, default=2.0, metavar="SECONDS",
                        help="Reject logging gaps inside accepted RTK intervals; quality exclusions are skipped. Default: 2 s; 0 disables.")
    parser.add_argument("--max-rtk-speed", type=float, default=100.0, metavar="METERS_PER_SECOND",
                        help="Reject resampled RTK steps above this speed before partitioning; default: 100 m/s; 0 disables.")
    parser.add_argument("--recursive", action="store_true", help="Find PCD files in subfolders too.")
    parser.add_argument("--drop-partial", action="store_true", help="Omit the single final remainder after accumulating all valid intervals.")
    parser.add_argument("--output-dir", type=Path, default=Path("comparison_results"),
                        help="Output directory; existing report files are overwritten.")
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError, csv.Error, InvalidOperation, OverflowError, ProjError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
