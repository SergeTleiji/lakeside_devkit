"""Top-level Lakeside dataset access."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .io import COMPONENTS, TRAVERSES, as_path
from .sequence import LakesideSequence


@dataclass(frozen=True)
class SequenceId:
    """Identifier discovered from root-level component folders."""

    date: str
    traverse: str


class LakesideDataset:
    """Discover and open Lakeside dataset sequences.

    Parameters
    ----------
    root:
        Dataset root containing folders named ``{date}_{traverse}_{component}``.
    """

    def __init__(self, root: str | Path):
        self.root = as_path(root)

    def discover_sequences(self) -> list[SequenceId]:
        """Return sequence identifiers discovered under ``root``."""

        if not self.root.exists():
            return []

        discovered: set[tuple[str, str]] = set()
        for child in self.root.iterdir():
            if not child.is_dir():
                continue
            parts = child.name.split("_")
            if len(parts) < 3:
                continue
            date = parts[0]
            traverse = parts[1]
            component = "_".join(parts[2:])
            base_component = component.removesuffix("_downsampled")
            if traverse in TRAVERSES and base_component in COMPONENTS:
                discovered.add((date, traverse))

        return [SequenceId(date, traverse) for date, traverse in sorted(discovered)]

    def sequences(self, use_downsampled_images: bool = False) -> list[LakesideSequence]:
        """Return discovered sequences as ``LakesideSequence`` objects."""

        return [
            self.sequence(seq.date, seq.traverse, use_downsampled_images)
            for seq in self.discover_sequences()
        ]

    def sequence(
        self, date: str, traverse: str, use_downsampled_images: bool = False
    ) -> LakesideSequence:
        """Open one sequence by recording date and traversal type."""

        return LakesideSequence(
            self.root,
            date=date,
            traverse=traverse,
            use_downsampled_images=use_downsampled_images,
        )
