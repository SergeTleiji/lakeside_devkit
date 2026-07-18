"""Coordinate and pose transform utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Transform:
    """Named rigid transform container."""

    source_frame: str
    target_frame: str
    matrix: tuple[tuple[float, float, float, float], ...]


def latlon_to_enu(
    lat,
    lon,
    alt,
    origin_lat=None,
    origin_lon=None,
    origin_alt=None,
):
    """Convert WGS84 latitude/longitude/altitude to local ENU coordinates.

    The first valid coordinate is used as the origin when one is not supplied.
    Returns an ``(N, 3)`` numpy array for array-like inputs, or a length-3 array
    for scalar inputs.
    """

    try:
        from pyproj import Transformer
    except ModuleNotFoundError as error:
        raise ImportError(
            "latlon_to_enu requires pyproj. Install it with: pip install pyproj"
        ) from error

    lat_array = np.asarray(lat, dtype=float)
    lon_array = np.asarray(lon, dtype=float)
    alt_array = np.asarray(alt, dtype=float)
    scalar_input = lat_array.ndim == 0

    lat_flat = np.atleast_1d(lat_array).astype(float)
    lon_flat = np.atleast_1d(lon_array).astype(float)
    alt_flat = np.atleast_1d(alt_array).astype(float)

    valid = np.isfinite(lat_flat) & np.isfinite(lon_flat) & np.isfinite(alt_flat)
    if not np.any(valid):
        raise ValueError("Cannot convert coordinates; no finite lat/lon/alt rows found.")

    if origin_lat is None:
        origin_lat = float(lat_flat[valid][0])
    if origin_lon is None:
        origin_lon = float(lon_flat[valid][0])
    if origin_alt is None:
        origin_alt = float(alt_flat[valid][0])

    ecef_transformer = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    x, y, z = ecef_transformer.transform(lon_flat, lat_flat, alt_flat)
    origin_x, origin_y, origin_z = ecef_transformer.transform(
        origin_lon, origin_lat, origin_alt
    )

    dx = np.asarray(x, dtype=float) - origin_x
    dy = np.asarray(y, dtype=float) - origin_y
    dz = np.asarray(z, dtype=float) - origin_z

    lat0 = math.radians(float(origin_lat))
    lon0 = math.radians(float(origin_lon))
    sin_lat = math.sin(lat0)
    cos_lat = math.cos(lat0)
    sin_lon = math.sin(lon0)
    cos_lon = math.cos(lon0)

    east = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    enu = np.column_stack((east, north, up))
    return enu[0] if scalar_input else enu


def normalize_quat(q: Iterable[float]) -> np.ndarray:
    """Return a unit quaternion as ``[qx, qy, qz, qw]``."""

    quat = np.asarray(tuple(q), dtype=float)
    if quat.shape != (4,):
        raise ValueError("Quaternion must have exactly four values: qx, qy, qz, qw.")
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero-length quaternion.")
    return quat / norm


def quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert a quaternion to a 3x3 rotation matrix."""

    x, y, z, w = normalize_quat((qx, qy, qz, qw))
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def make_transform(R, t) -> np.ndarray:
    """Create a 4x4 homogeneous transform from rotation and translation."""

    rotation = np.asarray(R, dtype=float)
    translation = np.asarray(t, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("Rotation must be a 3x3 matrix.")
    if translation.shape != (3,):
        raise ValueError("Translation must have exactly three values.")

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def invert_transform(T) -> np.ndarray:
    """Invert a 4x4 rigid homogeneous transform."""

    transform = np.asarray(T, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError("Transform must be a 4x4 matrix.")

    inverse = np.eye(4, dtype=float)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def identity_transform(source_frame: str, target_frame: str) -> Transform:
    """Return a named 4x4 identity transform."""

    matrix = tuple(tuple(float(value) for value in row) for row in np.eye(4))
    return Transform(source_frame=source_frame, target_frame=target_frame, matrix=matrix)
