#!/usr/bin/env python3
"""Render cam1/cam2/cam3 frames as a horizontal triplet."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lakeside import LakesideSequence, load_image  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Display or save cam1/cam2/cam3 images at the same sorted index."
    )
    parser.add_argument("--root", required=True, type=Path, help="Dataset root.")
    parser.add_argument("--date", required=True, help="YYMMDD recording date.")
    parser.add_argument(
        "--traverse",
        required=True,
        choices=("on-foot", "scooter"),
        help="Traversal type.",
    )
    parser.add_argument(
        "--index",
        required=True,
        type=int,
        help="Zero-based sorted frame index to load from each camera.",
    )
    parser.add_argument(
        "--downsampled",
        action="store_true",
        help="Prefer *_downsampled camera folders when available.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to outputs/{date}_{traverse}_triplet_{index}.png.",
    )
    parser.add_argument("--show", action="store_true", help="Display the triplet.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    plt = import_pyplot(show=args.show)

    sequence = LakesideSequence(
        args.root,
        args.date,
        args.traverse,
        use_downsampled_images=args.downsampled,
    )
    cameras = ("cam1", "cam2", "cam3")
    frames = []
    images = []
    for camera in cameras:
        try:
            frame = sequence.get_camera(camera, args.index)
        except IndexError as error:
            raise SystemExit(
                f"No {camera} frame at sorted index {args.index}."
            ) from error
        frames.append(frame)
        images.append(load_image(frame))

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, camera, frame, image in zip(axes, cameras, frames, images):
        axis.imshow(image)
        axis.set_title(f"{camera} frame {frame.frame_id}")
        axis.axis("off")
    figure.tight_layout()

    out_path = args.out or Path("outputs") / (
        f"{args.date}_{args.traverse}_triplet_{args.index}.png"
    )
    save_figure(figure, out_path)
    print(f"saved: {out_path}")

    if args.show:
        plt.show()
    plt.close(figure)
    return 0


def import_pyplot(show: bool):
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lakeside_matplotlib")
    )
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_figure(figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)


if __name__ == "__main__":
    raise SystemExit(main())
