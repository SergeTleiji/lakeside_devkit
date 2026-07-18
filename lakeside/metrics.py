"""Metrics and IMU CSV loading."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .transforms import latlon_to_enu


IMU_COLUMNS = [
    "timeStamp",
    "Orientation_x",
    "Orientation_y",
    "Orientation_z",
    "Orientation_w",
    "angular_vel_x",
    "angular_vel_y",
    "angular_vel_z",
    "linear_acc_x",
    "linear_acc_y",
    "linear_acc_z",
    "magnetic_x",
    "magnetic_y",
    "magnetic_z",
    "pressure",
    "temperature",
    "numSat",
    "latGPS",
    "longGPS",
    "altGPS",
    "insStatus",
    "latINS",
    "longINS",
    "altINS",
    "GPS_x",
    "GPS_y",
    "GPS_z",
    "INS_x",
    "INS_y",
    "INS_z",
    "RTK_Status",
    "latRTK",
    "longRTK",
    "altRTK",
]

RTK_COLUMNS = ["timeStamp", "latRTK", "longRTK", "altRTK", "RTK_Status"]
INS_COLUMNS = ["timeStamp", "latINS", "longINS", "altINS", "INS_x", "INS_y", "INS_z"]


def load_metrics_csv(path: Path) -> pd.DataFrame:
    """Load an ``Imu.csv`` file into a normalized pandas DataFrame.

    Header rows are detected automatically. Header names are stripped of leading
    ``%`` markers and whitespace. Headerless files receive the fixed column
    ordering defined by the dataset paper.
    """

    has_header = _has_header(path)
    if has_header:
        dataframe = pd.read_csv(path)
        dataframe.columns = [_clean_column_name(column) for column in dataframe.columns]
    else:
        dataframe = pd.read_csv(path, header=None, names=IMU_COLUMNS)

    dataframe = dataframe.rename(columns={"%timeStamp": "timeStamp"})
    dataframe = _coerce_numeric(dataframe)
    return dataframe


@dataclass
class MetricsData:
    """Convenience wrapper around a Lakeside metrics DataFrame."""

    dataframe: pd.DataFrame

    def timestamps(self) -> pd.Series:
        """Return metric timestamps."""

        return self.dataframe["timeStamp"]

    def rtk_status_counts(self) -> pd.Series:
        """Return RTK status value counts sorted by status."""

        if "RTK_Status" not in self.dataframe:
            return pd.Series(dtype="int64")
        return self.dataframe["RTK_Status"].value_counts(dropna=False).sort_index()

    def rtk_status_percentages(self) -> pd.Series:
        """Return RTK status percentages sorted by status."""

        counts = self.rtk_status_counts()
        if counts.empty:
            return pd.Series(dtype="float64")
        return counts / len(self.dataframe) * 100.0

    def filter_by_rtk_status(self, min_status: int = 3) -> "MetricsData":
        """Return rows whose RTK status is at least ``min_status``."""

        if "RTK_Status" not in self.dataframe:
            return MetricsData(self.dataframe.iloc[0:0].copy())
        mask = self.dataframe["RTK_Status"] >= min_status
        return MetricsData(self.dataframe.loc[mask].copy())

    def get_nearest_row(self, timestamp: float) -> pd.Series | None:
        """Return the row with the closest ``timeStamp`` value."""

        if self.dataframe.empty:
            return None
        deltas = (self.dataframe["timeStamp"] - timestamp).abs()
        return self.dataframe.loc[deltas.idxmin()]

    def get_rtk_positions(self, min_status: int | None = None) -> pd.DataFrame:
        """Return RTK position columns, optionally filtered by minimum status."""

        dataframe = self.dataframe
        if min_status is not None and "RTK_Status" in dataframe:
            dataframe = dataframe.loc[dataframe["RTK_Status"] >= min_status]
        return dataframe.loc[:, _existing_columns(dataframe, RTK_COLUMNS)].copy()

    def get_ins_positions(self) -> pd.DataFrame:
        """Return INS position columns."""

        return self.dataframe.loc[:, _existing_columns(self.dataframe, INS_COLUMNS)].copy()

    def rtk_to_enu(
        self, min_status: int = 3, origin: tuple[float, float, float] | None = None
    ) -> pd.DataFrame:
        """Return RTK positions converted to local ENU coordinates."""

        required = ["latRTK", "longRTK", "altRTK"]
        missing = [column for column in required if column not in self.dataframe]
        if missing:
            raise ValueError(f"Missing RTK coordinate columns: {', '.join(missing)}")

        dataframe = self.dataframe
        if "RTK_Status" in dataframe:
            dataframe = dataframe.loc[dataframe["RTK_Status"] >= min_status]
        dataframe = dataframe.dropna(subset=required).copy()
        if dataframe.empty:
            return dataframe.assign(enu_x=[], enu_y=[], enu_z=[])
        origin_lat, origin_lon, origin_alt = _resolve_origin(origin)
        enu = latlon_to_enu(
            dataframe["latRTK"].to_numpy(),
            dataframe["longRTK"].to_numpy(),
            dataframe["altRTK"].to_numpy(),
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            origin_alt=origin_alt,
        )
        result = dataframe.copy()
        result["enu_x"] = enu[:, 0]
        result["enu_y"] = enu[:, 1]
        result["enu_z"] = enu[:, 2]
        return result

    def ins_to_enu(
        self, origin: tuple[float, float, float] | None = None
    ) -> pd.DataFrame:
        """Return INS positions converted to local ENU coordinates."""

        required = ["latINS", "longINS", "altINS"]
        missing = [column for column in required if column not in self.dataframe]
        if missing:
            raise ValueError(f"Missing INS coordinate columns: {', '.join(missing)}")

        dataframe = self.dataframe.dropna(subset=required).copy()
        if dataframe.empty:
            return dataframe.assign(enu_x=[], enu_y=[], enu_z=[])
        origin_lat, origin_lon, origin_alt = _resolve_origin(origin)
        enu = latlon_to_enu(
            dataframe["latINS"].to_numpy(),
            dataframe["longINS"].to_numpy(),
            dataframe["altINS"].to_numpy(),
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            origin_alt=origin_alt,
        )
        dataframe["enu_x"] = enu[:, 0]
        dataframe["enu_y"] = enu[:, 1]
        dataframe["enu_z"] = enu[:, 2]
        return dataframe

    def summary(self) -> dict[str, Any]:
        """Return row count, timestamp range, and RTK status statistics."""

        timestamps = self.timestamps()
        counts = self.rtk_status_counts()
        percentages = self.rtk_status_percentages()
        if "RTK_Status" in self.dataframe:
            valid_fix_rows = int((self.dataframe["RTK_Status"] == 3).sum())
            float_or_fix_rows = int((self.dataframe["RTK_Status"] >= 2).sum())
        else:
            valid_fix_rows = 0
            float_or_fix_rows = 0

        return {
            "rows": len(self.dataframe),
            "timestamp_start": None if timestamps.empty else float(timestamps.min()),
            "timestamp_end": None if timestamps.empty else float(timestamps.max()),
            "rtk_status_counts": counts,
            "rtk_status_percentages": percentages,
            "rtk_fix_rows": valid_fix_rows,
            "rtk_float_or_fix_rows": float_or_fix_rows,
        }


def _has_header(path: Path) -> bool:
    with path.open("r", newline="") as stream:
        for row in csv.reader(stream):
            if not row:
                continue
            return any(not _is_number(cell.strip().lstrip("%")) for cell in row)
    return False


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _clean_column_name(column: object) -> str:
    return str(column).strip().lstrip("%").strip()


def _coerce_numeric(dataframe: pd.DataFrame) -> pd.DataFrame:
    for column in dataframe.columns:
        series = dataframe[column]
        converted = pd.to_numeric(series, errors="coerce")
        if converted.notna().sum() == series.notna().sum():
            dataframe[column] = converted
    return dataframe


def _existing_columns(dataframe: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in dataframe.columns]


def _resolve_origin(
    origin: tuple[float, float, float] | None,
) -> tuple[float | None, float | None, float | None]:
    if origin is None:
        return None, None, None
    if len(origin) != 3:
        raise ValueError("Origin must be a tuple of (lat, lon, alt).")
    return origin
