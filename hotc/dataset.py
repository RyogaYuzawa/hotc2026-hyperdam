"""HOTC validation dataset loading."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


VALIDATION_MODALITIES = {
    "nir": "HSI-NIR-FalseColor",
    "rednir": "HSI-RedNIR-FalseColor",
    "vis": "HSI-VIS-FalseColor",
}


class DatasetError(RuntimeError):
    pass


@dataclass(frozen=True)
class Sequence:
    name: str
    ids: tuple[str, ...]
    frames: tuple[Path, ...]
    boxes: tuple[tuple[int, int, int, int], ...] | None


def split_id(identifier: str) -> tuple[str, int]:
    sequence, separator, frame_text = identifier.rpartition("_")
    if not separator or not sequence or not frame_text.isdigit():
        raise DatasetError(f"invalid HOTC ID: {identifier!r}")
    frame = int(frame_text)
    if frame < 1:
        raise DatasetError(f"invalid HOTC frame: {identifier!r}")
    return sequence, frame


def read_bbox_csv(
    path: Path,
) -> dict[str, list[tuple[str, int, tuple[int, int, int, int]]]]:
    grouped: dict[
        str, list[tuple[str, int, tuple[int, int, int, int]]]
    ] = defaultdict(list)
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["ID", "x", "y", "width", "height"]:
            raise DatasetError(f"{path}: unexpected columns {reader.fieldnames}")
        for row in reader:
            sequence, frame = split_id(row["ID"])
            try:
                box = tuple(int(row[key]) for key in ("x", "y", "width", "height"))
            except ValueError as error:
                raise DatasetError(f"{path}: non-integer bbox for {row['ID']}") from error
            grouped[sequence].append((row["ID"], frame, box))
    for sequence, rows in grouped.items():
        rows.sort(key=lambda item: item[1])
        actual = [row[1] for row in rows]
        expected = list(range(actual[0], actual[0] + len(rows)))
        if actual != expected:
            raise DatasetError(f"{sequence}: CSV frame numbers are not contiguous")
    return dict(grouped)


def _frame_directory(
    root: Path, sequence: str, expected_count: int
) -> Path:
    prefix, separator, short_name = sequence.partition("-")
    if not separator or prefix not in VALIDATION_MODALITIES or not short_name:
        raise DatasetError(f"unknown sequence prefix: {sequence!r}")
    candidates = [Path(root) / VALIDATION_MODALITIES[prefix] / short_name]
    inspected: list[Path] = []
    for candidate in candidates:
        if not candidate.is_dir():
            inspected.append(candidate)
            continue
        directories = [
            candidate,
            *sorted(path for path in candidate.rglob("*") if path.is_dir()),
        ]
        matching = [
            directory
            for directory in directories
            if sum(
                path.suffix.lower() in {".jpg", ".jpeg"}
                for path in directory.iterdir()
            )
            == expected_count
        ]
        if len(matching) == 1:
            return matching[0]
        inspected.extend(directories)
    raise DatasetError(
        f"{sequence}: no directory with {expected_count} JPEGs under {inspected}"
    )


def _frames(
    root: Path, sequence: str, expected_count: int
) -> tuple[Path, ...]:
    directory = _frame_directory(root, sequence, expected_count)
    frames = sorted(
        (
            path
            for path in directory.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg"}
        ),
        key=lambda path: int(path.stem) if path.stem.isdigit() else -1,
    )
    if not frames or any(not path.stem.isdigit() for path in frames):
        raise DatasetError(f"{sequence}: JPEG names must be numeric")
    return tuple(frames)


def load_validation_sequences(
    root: Path,
    sample: Path,
    init_boxes: dict[str, tuple[int, int, int, int]],
) -> list[Sequence]:
    grouped = read_bbox_csv(sample)
    sequences: list[Sequence] = []
    for name, rows in sorted(grouped.items()):
        frames = _frames(root, name, expected_count=len(rows))
        if len(frames) != len(rows):
            raise DatasetError(
                f"{name}: {len(frames)} JPEGs != {len(rows)} sample rows"
            )
        if name not in init_boxes:
            raise DatasetError(f"{name}: initial bbox not found")
        boxes = (init_boxes[name],) + tuple(rows[index][2] for index in range(1, len(rows)))
        sequences.append(
            Sequence(
                name=name,
                ids=tuple(row[0] for row in rows),
                frames=frames,
                boxes=boxes,
            )
        )
    return sequences
