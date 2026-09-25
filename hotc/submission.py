"""Submission generation and validation for HOTC2026."""

from __future__ import annotations

import csv
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

SUBMISSION_COLUMNS = ("ID", "x", "y", "width", "height")
INTEGER_PATTERN = re.compile(r"-?[0-9]+")
SENSOR_PREFIXES = {
    "HSI-NIR": "nir",
    "HSI-RedNIR": "rednir",
    "HSI-VIS": "vis",
}


class SubmissionError(ValueError):
    """Raised when a submission violates the competition data contract."""


@dataclass(frozen=True)
class SubmissionSummary:
    rows: int
    sequences: int
    minimum_bbox: tuple[int, int, int, int]
    maximum_bbox: tuple[int, int, int, int]


def split_identifier(identifier: str) -> tuple[str, int]:
    sequence, separator, frame_text = identifier.rpartition("_")
    if not separator or not sequence or not frame_text.isdigit():
        raise SubmissionError(
            f"invalid ID {identifier!r}; expected <sequence>_<positive frame>"
        )
    frame = int(frame_text)
    if frame < 1:
        raise SubmissionError(f"invalid frame index in ID {identifier!r}")
    return sequence, frame


def read_sample_ids(sample_path: Path) -> list[str]:
    sample_path = Path(sample_path)
    if not sample_path.is_file():
        raise SubmissionError(f"sample submission not found: {sample_path}")

    with sample_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SUBMISSION_COLUMNS:
            raise SubmissionError(
                f"{sample_path}: expected columns {SUBMISSION_COLUMNS}, "
                f"got {reader.fieldnames}"
            )
        identifiers = [row["ID"] for row in reader]

    if not identifiers:
        raise SubmissionError(f"{sample_path}: sample submission is empty")
    if len(set(identifiers)) != len(identifiers):
        raise SubmissionError(f"{sample_path}: sample contains duplicate IDs")
    for identifier in identifiers:
        split_identifier(identifier)
    return identifiers


def _parse_integer(value: str, *, row_number: int, column: str) -> int:
    if not INTEGER_PATTERN.fullmatch(value):
        raise SubmissionError(
            f"row {row_number}: {column} must be an integer, got {value!r}"
        )
    return int(value)


def validate_submission(
    submission_path: Path,
    sample_path: Path,
) -> SubmissionSummary:
    expected_ids = read_sample_ids(sample_path)
    submission_path = Path(submission_path)
    if submission_path.suffix.lower() != ".csv":
        raise SubmissionError(f"submission must be a .csv file: {submission_path}")
    if not submission_path.is_file():
        raise SubmissionError(f"submission not found: {submission_path}")

    actual_ids: list[str] = []
    boxes: list[tuple[int, int, int, int]] = []
    with submission_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SUBMISSION_COLUMNS:
            raise SubmissionError(
                f"{submission_path}: expected columns {SUBMISSION_COLUMNS}, "
                f"got {reader.fieldnames}"
            )
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise SubmissionError(f"row {row_number}: malformed CSV row")
            identifier = row["ID"]
            split_identifier(identifier)
            box = tuple(
                _parse_integer(row[column], row_number=row_number, column=column)
                for column in SUBMISSION_COLUMNS[1:]
            )
            x, y, width, height = box
            if x < 0 or y < 0:
                raise SubmissionError(
                    f"row {row_number}: x and y must be non-negative, got {box}"
                )
            if width <= 0 or height <= 0:
                raise SubmissionError(
                    f"row {row_number}: width and height must be positive, got {box}"
                )
            actual_ids.append(identifier)
            boxes.append(box)

    if len(actual_ids) != len(expected_ids):
        raise SubmissionError(
            f"row count mismatch: expected {len(expected_ids)}, got {len(actual_ids)}"
        )
    if actual_ids != expected_ids:
        mismatch = next(
            index
            for index, (actual, expected) in enumerate(
                zip(actual_ids, expected_ids), start=2
            )
            if actual != expected
        )
        raise SubmissionError(
            f"ID/order mismatch at CSV row {mismatch}: "
            f"expected {expected_ids[mismatch - 2]!r}, "
            f"got {actual_ids[mismatch - 2]!r}"
        )

    sequences = {split_identifier(identifier)[0] for identifier in actual_ids}
    columns = tuple(zip(*boxes))
    minimum = tuple(min(column) for column in columns)
    maximum = tuple(max(column) for column in columns)
    return SubmissionSummary(
        rows=len(actual_ids),
        sequences=len(sequences),
        minimum_bbox=minimum,
        maximum_bbox=maximum,
    )


def _parse_init_rect(path: Path) -> tuple[int, int, int, int]:
    text = path.read_text(encoding="utf-8-sig").strip()
    fields = [field.strip() for field in re.split(r"[\s,;]+", text) if field.strip()]
    if len(fields) != 4:
        raise SubmissionError(f"{path}: expected four bbox values, got {fields}")

    values: list[int] = []
    for field in fields:
        try:
            number = float(field)
        except ValueError as error:
            raise SubmissionError(f"{path}: invalid bbox value {field!r}") from error
        if not math.isfinite(number):
            raise SubmissionError(f"{path}: non-finite bbox value {field!r}")
        values.append(int(round(number)))

    box = tuple(values)
    x, y, width, height = box
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise SubmissionError(f"{path}: invalid bbox {box}")
    return box


def discover_init_bboxes(
    official_root: Path,
    required_sequences: Iterable[str],
) -> dict[str, tuple[int, int, int, int]]:
    official_root = Path(official_root)
    if not official_root.is_dir():
        raise SubmissionError(f"official data root not found: {official_root}")

    required = set(required_sequences)
    candidates: dict[str, list[tuple[Path, tuple[int, int, int, int]]]] = {}
    for path in official_root.rglob("init_rect.txt"):
        sensor_prefix = next(
            (
                prefix
                for directory, prefix in SENSOR_PREFIXES.items()
                if directory in path.parts
                or f"{directory}-FalseColor" in path.parts
            ),
            None,
        )
        sequence = (
            f"{sensor_prefix}-{path.parent.name}"
            if sensor_prefix is not None
            else path.parent.name
        )
        if sequence in required:
            candidates.setdefault(sequence, []).append((path, _parse_init_rect(path)))

    missing = sorted(required - candidates.keys())
    if missing:
        preview = ", ".join(missing[:10])
        suffix = " ..." if len(missing) > 10 else ""
        raise SubmissionError(
            f"missing init_rect.txt for {len(missing)} sequences: {preview}{suffix}"
        )

    resolved: dict[str, tuple[int, int, int, int]] = {}
    for sequence, entries in candidates.items():
        unique_boxes = {box for _, box in entries}
        if len(unique_boxes) != 1:
            details = ", ".join(f"{path}={box}" for path, box in entries)
            raise SubmissionError(
                f"conflicting init_rect.txt values for {sequence}: {details}"
            )
        resolved[sequence] = unique_boxes.pop()
    return resolved


def write_submission(
    sample_path: Path,
    output_path: Path,
    predictions: Mapping[str, Sequence[int]],
) -> SubmissionSummary:
    sample_path = Path(sample_path)
    output_path = Path(output_path)
    expected_ids = read_sample_ids(sample_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, int, int, int, int]] = []
    for identifier in expected_ids:
        if identifier not in predictions:
            raise SubmissionError(f"no prediction for ID {identifier!r}")
        raw_box = predictions[identifier]
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in raw_box
        ):
            raise SubmissionError(
                f"prediction for {identifier!r} must contain integers: {raw_box}"
            )
        box = tuple(raw_box)
        if len(box) != 4:
            raise SubmissionError(f"prediction for {identifier!r} is not length four")
        x, y, width, height = box
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise SubmissionError(f"invalid prediction for {identifier!r}: {box}")
        rows.append((identifier, x, y, width, height))

    extra = set(predictions) - set(expected_ids)
    if extra:
        preview = ", ".join(sorted(extra)[:10])
        raise SubmissionError(f"predictions contain unexpected IDs: {preview}")

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(SUBMISSION_COLUMNS)
            writer.writerows(rows)
        os.replace(temporary_name, output_path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)

    return validate_submission(output_path, sample_path)
