"""Sequence-level access to Lakeside sensor files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import (
    CAMERAS,
    COMPONENTS,
    IMAGE_EXTENSIONS,
    TRAVERSES,
    as_path,
    component_folder_name,
    sorted_files,
)
from .metrics import MetricsData, load_metrics_csv
from .sensors import CameraFrame, LidarFrame


@dataclass(frozen=True)
class SensorFileSummary:
    """Compact listing summary for one sensor stream."""

    count: int
    first: Path | None
    last: Path | None


class LakesideSequence:
    """Access files for one Lakeside traversal.

    Parameters
    ----------
    root:
        Dataset root.
    date:
        YYMMDD-style recording date, such as ``200803``.
    traverse:
        Traversal type, currently ``on-foot`` or ``scooter``.
    use_downsampled_images:
        Prefer ``*_downsampled`` camera folders when available.
    """

    def __init__(
        self,
        root: str | Path,
        date: str,
        traverse: str,
        use_downsampled_images: bool = False,
    ):
        self.root = as_path(root)
        self.date = date
        self.traverse = traverse
        self.use_downsampled_images = use_downsampled_images

        if self.traverse not in TRAVERSES:
            allowed = ", ".join(TRAVERSES)
            raise ValueError(f"Unknown traverse '{traverse}'. Expected one of: {allowed}")

    def get_component_path(self, component: str) -> Path:
        """Return the best matching root-level folder for a component.

        The canonical folder is ``{date}_{traverse}_{component}``. For cameras,
        root-level ``*_downsampled`` component folders are also tolerated. The
        ``use_downsampled_images`` flag controls which candidate is preferred.
        """

        self._validate_component(component)
        canonical = self.root / component_folder_name(self.date, self.traverse, component)
        candidates = [canonical]

        if component in CAMERAS:
            downsampled = self.root / component_folder_name(
                self.date, self.traverse, f"{component}_downsampled"
            )
            candidates = [downsampled, canonical] if self.use_downsampled_images else [
                canonical,
                downsampled,
            ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return candidates[0]

    def lidar_files(self) -> list[Path]:
        """Return lidar ``.pcd`` files sorted by timestamp-like filename stem."""

        lidar_dir = self.get_component_path("lidar") / self.date
        if not lidar_dir.exists():
            return []
        return sorted_files(path for path in lidar_dir.glob("*.pcd") if path.is_file())

    def iter_lidar(self):
        """Yield lidar frames in timestamp order."""

        for index, path in enumerate(self.lidar_files()):
            yield LidarFrame(path=path, timestamp=self._parse_float_stem(path), index=index)

    def lidar_timestamps(self) -> list[float]:
        """Return lidar Unix timestamps parsed from PCD filename stems."""

        return [frame.timestamp for frame in self.iter_lidar()]

    def camera_files(self, camera: str = "cam1") -> list[Path]:
        """Return image files for a camera, sorted numerically by filename stem."""

        self._validate_camera(camera)
        image_dir = self._camera_image_dir(camera)
        if image_dir is None or not image_dir.exists():
            return []

        return sorted_files(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    def iter_camera(self, camera: str):
        """Yield camera frames in numeric filename order."""

        # TODO: Add camera timestamps if a documented timestamp file is found.
        # Camera filename stems are frame IDs, not Unix timestamps.
        self._validate_camera(camera)
        for index, path in enumerate(self.camera_files(camera)):
            yield CameraFrame(
                path=path,
                camera=camera,
                frame_id=self._parse_int_stem(path),
                index=index,
            )

    def get_lidar(self, index: int) -> LidarFrame:
        """Return one lidar frame by zero-based sequence index."""

        return list(self.iter_lidar())[index]

    def get_camera(self, camera: str, index: int) -> CameraFrame:
        """Return one camera frame by zero-based sequence index."""

        return list(self.iter_camera(camera))[index]

    def get_lidar_by_timestamp(
        self, timestamp: float, nearest: bool = True
    ) -> LidarFrame | None:
        """Return a lidar frame by timestamp, optionally nearest-neighbor."""

        frames = list(self.iter_lidar())
        if not frames:
            return None
        if nearest:
            return min(frames, key=lambda frame: abs(frame.timestamp - timestamp))
        for frame in frames:
            if frame.timestamp == timestamp:
                return frame
        return None

    def get_camera_by_frame_id(
        self, camera: str, frame_id: int, nearest: bool = True
    ) -> CameraFrame | None:
        """Return a camera frame by filename stem, optionally nearest-neighbor."""

        frames = list(self.iter_camera(camera))
        if not frames:
            return None
        if nearest:
            return min(frames, key=lambda frame: abs(frame.frame_id - frame_id))
        for frame in frames:
            if frame.frame_id == frame_id:
                return frame
        return None

    def metrics_file(self) -> Path | None:
        """Return the IMU metrics CSV path if it exists."""

        imu_file = self.get_component_path("metrics") / self.date / "Imu.csv"
        return imu_file if imu_file.exists() else None

    def load_metrics(self) -> MetricsData:
        """Load this sequence's ``Imu.csv`` metrics data."""

        metrics_file = self.metrics_file()
        if metrics_file is None:
            raise FileNotFoundError(
                f"Could not find Imu.csv for {self.date} {self.traverse}"
            )
        return MetricsData(load_metrics_csv(metrics_file))

    def metrics_summary(self) -> dict[str, Any]:
        """Return summary statistics for this sequence's metrics data."""

        return self.load_metrics().summary()

    def get_nearest_metrics(self, timestamp: float):
        """Return the metrics row nearest to ``timestamp``."""

        return self.load_metrics().get_nearest_row(timestamp)

    def metrics_timestamps(self) -> list[float]:
        """Return metrics Unix timestamps from ``Imu.csv``."""

        return self.load_metrics().timestamps().astype(float).tolist()

    def get_metrics_for_lidar(self, lidar_index: int):
        """Return the nearest metrics row for one lidar frame index."""

        from .sync import match_lidar_to_metrics

        match = match_lidar_to_metrics(self, self.get_lidar(lidar_index))
        return None if match is None else match["metrics_row"]

    def summary(self) -> dict[str, Any]:
        """Return resolved paths, counts, and first/last files for the sequence."""

        sensors: dict[str, SensorFileSummary] = {
            "lidar": self._summarize_files(self.lidar_files())
        }
        for camera in CAMERAS:
            sensors[camera] = self._summarize_files(self.camera_files(camera))

        return {
            "root": self.root,
            "date": self.date,
            "traverse": self.traverse,
            "use_downsampled_images": self.use_downsampled_images,
            "component_paths": {
                component: self.get_component_path(component) for component in COMPONENTS
            },
            "sensors": sensors,
            "metrics_file": self.metrics_file(),
        }

    def _camera_image_dir(self, camera: str) -> Path | None:
        component_path = self.get_component_path(camera)
        preferred_names = (
            [f"{camera}_downsampled", camera]
            if self.use_downsampled_images
            else [camera, f"{camera}_downsampled"]
        )

        for dirname in preferred_names:
            candidate = component_path / dirname
            if candidate.exists():
                return candidate

        if component_path.exists():
            for child in sorted(component_path.iterdir()):
                if child.is_dir() and child.name in preferred_names:
                    return child
        return None

    @staticmethod
    def _summarize_files(files: list[Path]) -> SensorFileSummary:
        return SensorFileSummary(
            count=len(files),
            first=files[0] if files else None,
            last=files[-1] if files else None,
        )

    @staticmethod
    def _parse_float_stem(path: Path) -> float:
        try:
            return float(path.stem)
        except ValueError as error:
            raise ValueError(f"Expected numeric timestamp filename stem: {path}") from error

    @staticmethod
    def _parse_int_stem(path: Path) -> int:
        try:
            return int(path.stem)
        except ValueError as error:
            raise ValueError(f"Expected integer frame filename stem: {path}") from error

    @staticmethod
    def _validate_camera(camera: str) -> None:
        if camera not in CAMERAS:
            allowed = ", ".join(CAMERAS)
            raise ValueError(f"Unknown camera '{camera}'. Expected one of: {allowed}")

    @staticmethod
    def _validate_component(component: str) -> None:
        if component not in COMPONENTS:
            allowed = ", ".join(COMPONENTS)
            raise ValueError(f"Unknown component '{component}'. Expected one of: {allowed}")
