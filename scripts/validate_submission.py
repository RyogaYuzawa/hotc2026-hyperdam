#!/usr/bin/env python3
"""Validate a HOTC2026 submission against the official sample."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hotc.submission import SubmissionError, validate_submission


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument(
        "--sample",
        type=Path,
        default=Path("data/HOTC2026/sample_submisson.csv"),
    )
    args = parser.parse_args()

    try:
        summary = validate_submission(args.submission, args.sample)
    except SubmissionError as error:
        parser.exit(1, f"INVALID: {error}\n")

    print(
        "VALID: "
        f"rows={summary.rows:,}, sequences={summary.sequences:,}, "
        f"bbox_min={summary.minimum_bbox}, bbox_max={summary.maximum_bbox}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
