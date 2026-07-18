"""Filesystem helpers for Lakeside dataset discovery."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable


CAMERAS = ("cam1", "cam2", "cam3")
COMPONENTS = (*CAMERAS, "lidar", "metrics")
TRAVERSES = ("on-foot", "scooter")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def as_path(path: str | Path) -> Path:
    """Return a user-supplied path as an expanded ``Path``."""

    return Path(path).expanduser()


def component_folder_name(date: str, traverse: str, component: str) -> str:
    """Return the canonical root-level component folder name."""

    return f"{date}_{traverse}_{component}"


def numeric_stem_key(path: Path) -> tuple[int, Decimal | str, str]:
    """Sort paths by numeric filename stems when possible.

    Camera files use integer-like stems, while lidar PCD files use decimal Unix
    timestamps. Non-numeric stems are sorted after numeric stems by text.
    """

    try:
        return (0, Decimal(path.stem), path.name)
    except InvalidOperation:
        return (1, path.stem, path.name)


def sorted_files(files: Iterable[Path]) -> list[Path]:
    """Return files sorted with dataset-aware filename semantics."""

    return sorted(files, key=numeric_stem_key)


def load_pcd(path: str | Path, backend: str = "auto"):
    """Load PCD XYZ points as an ``(N, 3)`` numpy array.

    Parameters
    ----------
    path:
        PCD file path.
    backend:
        ``auto`` tries the internal ASCII parser first, then Open3D.
        ``ascii`` uses the internal parser only.
        ``open3d`` uses Open3D only.
    """

    fields = load_pcd_fields(path, backend=backend)
    return fields["xyz"]


def load_pcd_fields(path: str | Path, backend: str = "auto") -> dict:
    """Load available PCD fields.

    The returned dictionary always contains ``xyz`` when loading succeeds.
    Internal ASCII loading may also return ``intensity`` and common semantic
    columns. Open3D loading returns XYZ only.
    """

    path = as_path(path)
    backend = backend.lower()
    if backend not in {"auto", "ascii", "open3d"}:
        raise ValueError("PCD backend must be one of: auto, ascii, open3d")

    if backend == "ascii":
        return _load_ascii_pcd_fields(path)
    if backend == "open3d":
        return _load_open3d_pcd_fields(path)

    try:
        return _load_ascii_pcd_fields(path)
    except NotImplementedError:
        return _load_open3d_pcd_fields(path)


def _load_ascii_pcd_fields(path: Path) -> dict:
    import numpy as np

    with path.open("rb") as stream:
        fields: list[str] = []
        data_mode: str | None = None

        while True:
            line_bytes = stream.readline()
            if line_bytes == b"":
                break
            line = line_bytes.decode("ascii", errors="strict").strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            key = parts[0].upper()
            if key == "FIELDS":
                fields = parts[1:]
            elif key == "DATA":
                data_mode = parts[1].lower() if len(parts) > 1 else ""
                break

        if data_mode is None:
            raise ValueError(f"PCD file is missing a DATA header: {path}")
        if data_mode != "ascii":
            raise NotImplementedError(
                f"PCD data mode '{data_mode}' is not supported by the internal "
                f"ASCII parser for {path}."
            )
        if not fields:
            raise ValueError(f"PCD file is missing a FIELDS header: {path}")

        try:
            xyz_indices = [fields.index(axis) for axis in ("x", "y", "z")]
        except ValueError as error:
            raise ValueError(f"PCD file must contain x, y, z fields: {path}") from error

        optional_field_names = [
            name
            for name in (
                "intensity",
                "label",
                "semantic",
                "semantic_id",
                "class",
                "rgb",
            )
            if name in fields
        ]
        optional_indices = {name: fields.index(name) for name in optional_field_names}

        points: list[tuple[float, float, float]] = []
        optional_values: dict[str, list[float]] = {
            name: [] for name in optional_field_names
        }
        for line_bytes in stream:
            line = line_bytes.decode("ascii", errors="strict").strip()
            if not line:
                continue
            values = line.split()
            try:
                points.append(tuple(float(values[i]) for i in xyz_indices))
                for name, index in optional_indices.items():
                    optional_values[name].append(float(values[index]))
            except IndexError as error:
                raise ValueError(f"Malformed PCD point row in {path}: {line}") from error

    result = {"xyz": np.asarray(points, dtype=np.float32)}
    for name, values in optional_values.items():
        result[name] = np.asarray(values, dtype=np.float32)
    return result


def _load_open3d_pcd_fields(path: Path) -> dict:
    try:
        import numpy as np
        import open3d as o3d
    except ModuleNotFoundError as error:
        raise ImportError(
            "Open3D is required for this PCD format. Install with: pip install open3d"
        ) from error

    point_cloud = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(point_cloud.points, dtype=np.float32)
    if xyz.ndim != 2 or (xyz.size and xyz.shape[1] != 3):
        raise ValueError(f"Open3D did not return XYZ points for {path}")
    return {"xyz": xyz.reshape((-1, 3))}
