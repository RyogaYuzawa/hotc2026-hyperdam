#!/usr/bin/env python3
"""Create an inference ID CSV by scanning a HOTC-format validation directory."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hotc.submission import discover_init_bboxes  # noqa: E402


MODALITIES = {
    "HSI-NIR-FalseColor": "nir",
    "HSI-RedNIR-FalseColor": "rednir",
    "HSI-VIS-FalseColor": "vis",
}


def frame_count(sequence_root: Path) -> int:
    matches: list[tuple[Path, int]] = []
    for directory in (sequence_root, *sorted(sequence_root.rglob("*"))):
        if not directory.is_dir():
            continue
        frames = [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
        ]
        if frames and all(path.stem.isdigit() for path in frames):
            matches.append((directory, len(frames)))
    if len(matches) != 1:
        raise ValueError(
            f"{sequence_root}: expected exactly one numeric JPEG directory, "
            f"found {matches}"
        )
    return matches[0][1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    validation_root = args.validation_root.resolve()
    sequence_counts: dict[str, int] = {}
    for directory_name, prefix in MODALITIES.items():
        modality_root = validation_root / directory_name
        if not modality_root.is_dir():
            continue
        for sequence_root in sorted(path for path in modality_root.iterdir() if path.is_dir()):
            name = f"{prefix}-{sequence_root.name}"
            if name in sequence_counts:
                raise ValueError(f"duplicate sequence name: {name}")
            sequence_counts[name] = frame_count(sequence_root)

    if not sequence_counts:
        expected = ", ".join(MODALITIES)
        raise ValueError(
            f"no HOTC false-color sequences found under {validation_root}; "
            f"expected one of: {expected}"
        )

    initial_boxes = discover_init_bboxes(validation_root, sequence_counts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("ID", "x", "y", "width", "height"))
        for sequence in sorted(sequence_counts):
            initial_box = initial_boxes[sequence]
            for frame in range(1, sequence_counts[sequence] + 1):
                writer.writerow((f"{sequence}_{frame}", *initial_box))

    print(
        f"Generated sample CSV: sequences={len(sequence_counts)} "
        f"frames={sum(sequence_counts.values())} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
