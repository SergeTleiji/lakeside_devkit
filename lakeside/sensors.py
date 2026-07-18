"""Sensor frame containers and lightweight file loaders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .io import load_pcd


@dataclass
class LidarFrame:
    """One lidar point cloud frame."""

    path: Path
    timestamp: float
    index: int


@dataclass
class CameraFrame:
    """One camera image frame."""

    path: Path
    camera: str
    frame_id: int
    index: int


def load_image(camera_frame: CameraFrame):
    """Load a camera frame as an RGB numpy array.

    Pillow is preferred when available. If Pillow is not installed, OpenCV is
    tried and converted from BGR to RGB. A clear ImportError is raised when no
    supported image backend is installed.
    """

    try:
        import numpy as np
    except ModuleNotFoundError as error:
        raise ImportError("load_image requires numpy.") from error

    try:
        from PIL import Image

        with Image.open(camera_frame.path) as image:
            return np.asarray(image.convert("RGB"))
    except ModuleNotFoundError:
        try:
            import cv2
        except ModuleNotFoundError as error:
            raise ImportError(
                "load_image requires Pillow or OpenCV. Install pillow or "
                "opencv-python to load camera frames."
            ) from error

        image = cv2.imread(str(camera_frame.path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image file: {camera_frame.path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_pcd_xyz(path: Path, backend: str = "auto"):
    """Load x/y/z columns from a PCD file as a numpy array."""

    return load_pcd(path, backend=backend)
