#!/usr/bin/env python3
"""Apply amodal-v10 edge expansions to HSI-v3, then RTS-fill empty masks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hotc.amodal.smoothing import _rts_smooth  # noqa: E402
from hotc.submission import read_sample_ids, split_identifier, validate_submission  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--amodal-cache-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--sample",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument("--rts-size-scale", type=float, default=0.95)
    parser.add_argument("--center-acceleration-std", type=float, default=1.0)
    parser.add_argument("--size-acceleration-std", type=float, default=0.01)
    parser.add_argument("--center-measurement-ratio", type=float, default=0.12)
    parser.add_argument("--size-measurement-std", type=float, default=0.06)
    return parser.parse_args()


def rts_fill(
    boxes: np.ndarray,
    missing: np.ndarray,
    image_size: tuple[int, int],
    args: SimpleNamespace,
) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64)
    missing = np.asarray(missing, dtype=bool).copy()
    missing[0] = False
    size = np.maximum(boxes[:, 2:], 1.0)
    center = boxes[:, :2] + 0.5 * size
    observations = np.column_stack((center, np.log(size)))
    smoothed = _rts_smooth(
        observations,
        ~missing,
        center_acceleration_std=args.center_acceleration_std,
        size_acceleration_std=args.size_acceleration_std,
        center_measurement_ratio=args.center_measurement_ratio,
        size_measurement_std=args.size_measurement_std,
    )
    inferred_size = np.exp(smoothed[:, 2:]) * args.rts_size_scale
    inferred_center = smoothed[:, :2]
    width, height = image_size
    bounds = np.asarray([width, height], dtype=np.float64)
    inferred_size = np.minimum(np.maximum(inferred_size, 1.0), bounds)
    inferred_xy = np.minimum(
        np.maximum(inferred_center - 0.5 * inferred_size, 0.0),
        bounds - inferred_size,
    )
    inferred = np.column_stack((inferred_xy, inferred_size)).astype(np.float32)
    output = boxes.astype(np.float32).copy()
    output[missing] = inferred[missing]
    return output


def read_submission(path: Path) -> dict[str, tuple[int, int, int, int]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ("ID", "x", "y", "width", "height"):
            raise ValueError(f"unexpected columns in {path}: {reader.fieldnames}")
        rows = {
            row["ID"]: tuple(int(row[key]) for key in ("x", "y", "width", "height"))
            for row in reader
        }
    if len(rows) == 0:
        raise ValueError(f"empty submission: {path}")
    return rows


def clip_float_boxes(
    boxes: np.ndarray, image_size: tuple[int, int]
) -> np.ndarray:
    width, height = image_size
    result = np.asarray(boxes, dtype=np.float64).copy()
    result[:, 2:] = np.maximum(result[:, 2:], 1.0)
    result[:, 2] = np.minimum(result[:, 2], width)
    result[:, 3] = np.minimum(result[:, 3], height)
    result[:, 0] = np.clip(result[:, 0], 0.0, width - result[:, 2])
    result[:, 1] = np.clip(result[:, 1], 0.0, height - result[:, 3])
    return result.astype(np.float32)


def clip_and_round(
    boxes: np.ndarray, image_size: tuple[int, int]
) -> np.ndarray:
    width, height = image_size
    result = np.rint(boxes).astype(np.int64)
    result[:, 0] = np.clip(result[:, 0], 0, max(width - 1, 0))
    result[:, 1] = np.clip(result[:, 1], 0, max(height - 1, 0))
    result[:, 2] = np.clip(result[:, 2], 1, width - result[:, 0])
    result[:, 3] = np.clip(result[:, 3], 1, height - result[:, 1])
    return result


def load_cache(path: Path, expected_ids: list[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = {"ids", "image_size", "baseline", "head", "mask_nonempty"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path}: missing keys {sorted(missing)}")
        arrays = {key: data[key].copy() for key in required}
    ids = [str(value) for value in arrays["ids"].tolist()]
    if ids != expected_ids:
        raise ValueError(f"{path}: IDs do not match sample sequence")
    count = len(expected_ids)
    for key in ("baseline", "head"):
        if arrays[key].shape != (count, 4) or not np.isfinite(arrays[key]).all():
            raise ValueError(f"{path}: invalid {key}")
    if arrays["mask_nonempty"].shape != (count,):
        raise ValueError(f"{path}: invalid mask_nonempty")
    if arrays["image_size"].shape != (2,):
        raise ValueError(f"{path}: invalid image_size")
    return arrays


def edge_expansions(baseline: np.ndarray, head: np.ndarray) -> np.ndarray:
    baseline = np.asarray(baseline, dtype=np.float64)
    head = np.asarray(head, dtype=np.float64)
    baseline_right_bottom = baseline[:, :2] + baseline[:, 2:]
    head_right_bottom = head[:, :2] + head[:, 2:]
    expansion = np.column_stack(
        (
            baseline[:, 0] - head[:, 0],
            baseline[:, 1] - head[:, 1],
            head_right_bottom[:, 0] - baseline_right_bottom[:, 0],
            head_right_bottom[:, 1] - baseline_right_bottom[:, 1],
        )
    )
    # Float32 arithmetic can produce tiny negative residuals around zero.
    if (expansion < -1e-4).any():
        row, side = np.argwhere(expansion < -1e-4)[0]
        raise ValueError(
            f"amodal head shrinks side {side} at row {row}: {expansion[row, side]}"
        )
    return np.maximum(expansion, 0.0).astype(np.float32)


def apply_expansions(boxes: np.ndarray, expansion: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64)
    expansion = np.asarray(expansion, dtype=np.float64)
    result = boxes.copy()
    result[:, 0] -= expansion[:, 0]
    result[:, 1] -= expansion[:, 1]
    result[:, 2] += expansion[:, 0] + expansion[:, 2]
    result[:, 3] += expansion[:, 1] + expansion[:, 3]
    return result.astype(np.float32)


def main() -> int:
    args = parse_args()
    identifiers = read_sample_ids(args.sample)
    baseline_rows = read_submission(args.baseline)
    if set(baseline_rows) != set(identifiers):
        raise ValueError("sample and HSI-v3 IDs must match exactly")

    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for identifier in identifiers:
        sequence, frame = split_identifier(identifier)
        grouped[sequence].append((frame, identifier))

    rts_args = SimpleNamespace(
        rts_size_scale=args.rts_size_scale,
        center_acceleration_std=args.center_acceleration_std,
        size_acceleration_std=args.size_acceleration_std,
        center_measurement_ratio=args.center_measurement_ratio,
        size_measurement_std=args.size_measurement_std,
    )
    predictions: dict[str, np.ndarray] = {}
    diagnostics = {}
    for sequence in sorted(grouped):
        sequence_ids = [identifier for _, identifier in sorted(grouped[sequence])]
        hsi = np.asarray([baseline_rows[value] for value in sequence_ids], dtype=np.float32)
        arrays = load_cache(args.amodal_cache_root / f"{sequence}.npz", sequence_ids)
        image_size = tuple(int(value) for value in arrays["image_size"].tolist())
        nonempty = arrays["mask_nonempty"].astype(bool)
        missing = ~nonempty
        missing[0] = False

        expansion = edge_expansions(arrays["baseline"], arrays["head"])
        expansion[~nonempty] = 0.0
        expansion[0] = 0.0
        amodal = clip_float_boxes(apply_expansions(hsi, expansion), image_size)
        smoothed = rts_fill(amodal, missing, image_size, rts_args)
        output = clip_and_round(smoothed, image_size)
        hsi_int = clip_and_round(hsi, image_size)
        amodal_int = clip_and_round(amodal, image_size)
        output[0] = hsi_int[0]

        predictions[sequence] = output
        diagnostics[sequence] = {
            "frames": len(sequence_ids),
            "amodal_float_changed_frames": int(
                (nonempty & np.any(expansion > 1e-6, axis=1)).sum()
            ),
            "amodal_integer_changed_frames": int(
                np.any(amodal_int != hsi_int, axis=1).sum()
            ),
            "full_occlusion_routed_frames": int(missing.sum()),
            "rts_integer_changed_frames": int(
                np.any(output != amodal_int, axis=1).sum()
            ),
            "final_changed_from_hsi_frames": int(
                np.any(output != hsi_int, axis=1).sum()
            ),
            "frame_zero_preserved": bool(np.array_equal(output[0], hsi_int[0])),
        }

    initial_frame_violations = sum(
        not row["frame_zero_preserved"] for row in diagnostics.values()
    )
    if initial_frame_violations:
        raise RuntimeError(
            f"HSI composition changed frame zero in {initial_frame_violations} sequences"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("ID", "x", "y", "width", "height"))
        for identifier in identifiers:
            sequence, frame = split_identifier(identifier)
            writer.writerow((identifier, *predictions[sequence][frame - 1].tolist()))

    summary = validate_submission(args.output, args.sample)
    report = {
        "workflow": "hsi_v3_plus_amodal_v10_residual_empty_mask_rts",
        "contract": {
            "base_trajectory": "HSI DRM gate v3",
            "amodal_composition": (
                "transfer the cached head's nonnegative per-side expansion residual "
                "relative to its frozen CIE SAM box onto the HSI-v3 box"
            ),
            "amodal_scope": "nonempty SAM predicted-mask frames only",
            "rts_scope": "empty SAM predicted-mask frames only",
            "initial_prompt_hard_preserved": True,
            "offline_future_context": True,
        },
        "rows": summary.rows,
        "sequences": summary.sequences,
        "initial_frame_violations": initial_frame_violations,
        "amodal_float_changed_frames": sum(
            row["amodal_float_changed_frames"] for row in diagnostics.values()
        ),
        "amodal_integer_changed_frames": sum(
            row["amodal_integer_changed_frames"] for row in diagnostics.values()
        ),
        "full_occlusion_routed_frames": sum(
            row["full_occlusion_routed_frames"] for row in diagnostics.values()
        ),
        "rts_integer_changed_frames": sum(
            row["rts_integer_changed_frames"] for row in diagnostics.values()
        ),
        "final_changed_from_hsi_frames": sum(
            row["final_changed_from_hsi_frames"] for row in diagnostics.values()
        ),
        "rts": {
            "size_scale": args.rts_size_scale,
            "center_acceleration_std": args.center_acceleration_std,
            "size_acceleration_std": args.size_acceleration_std,
            "center_measurement_ratio": args.center_measurement_ratio,
            "size_measurement_std": args.size_measurement_std,
        },
        "baseline": str(args.baseline.resolve()),
        "amodal_cache_root": str(args.amodal_cache_root.resolve()),
        "amodal_model": "v10",
        "submission": str(args.output.resolve()),
        "sequence_diagnostics": diagnostics,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "sequence_diagnostics"},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
