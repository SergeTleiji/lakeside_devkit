"""Save a cam1/cam2/cam3 image triplet loaded by sorted frame index."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lakeside import LakesideSequence, load_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Save a Lakeside camera triplet preview.")
    parser.add_argument("--root", required=True, type=Path, help="Dataset root.")
    parser.add_argument("--date", default="200803", help="YYMMDD recording date.")
    parser.add_argument(
        "--traverse",
        default="on-foot",
        choices=("on-foot", "scooter"),
        help="Traversal type.",
    )
    parser.add_argument("--index", type=int, default=0, help="Sorted frame index.")
    parser.add_argument(
        "--downsampled",
        action="store_true",
        help="Prefer *_downsampled camera folders.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/image_triplet.png"),
        help="Output preview PNG.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    plt = import_pyplot()
    sequence = LakesideSequence(
        args.root,
        args.date,
        args.traverse,
        use_downsampled_images=args.downsampled,
    )

    cameras = ("cam1", "cam2", "cam3")
    frames = [sequence.get_camera(camera, args.index) for camera in cameras]
    images = [load_image(frame) for frame in frames]

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, camera, frame, image in zip(axes, cameras, frames, images):
        axis.imshow(image)
        axis.set_title(f"{camera} frame {frame.frame_id}")
        axis.axis("off")
    figure.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=150)
    plt.close(figure)
    print(f"saved: {args.out}")
    return 0


def import_pyplot():
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lakeside_matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


if __name__ == "__main__":
    raise SystemExit(main())
