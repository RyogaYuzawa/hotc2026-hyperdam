#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hotc.submission import (
    discover_init_bboxes,
    read_sample_ids,
    split_identifier,
    validate_submission,
    write_submission,
)
from hotc.config import load_config
from hotc.dataset import load_validation_sequences
from hotc.tracker import (
    _box_from_mask,
    _open_rgb,
    _release_sequence,
    prune_history,
)
from hotc.amodal.runtime import build_tracker as build_amodal_tracker
from hotc.amodal.inference import StreamingAmodalV10
from hotc.hsi.gate import ReferenceHSIIdentityGate
from hotc.hsi.hyperspectral import HyperspectralError, load_cube, raw_hsi_path
from hotc.hsi.static_scene import (
    StaticBBoxContainmentGate,
    StaticBBoxContainmentPolicyV3,
    box_mask,
    clip_box,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the HOTC 2026 HyperDAM inference pipeline."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--sequence", action="append")
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "hotc/config.toml",
    )
    parser.add_argument("--validation-root", type=Path)
    parser.add_argument("--sample", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "result/tracking",
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=PROJECT_ROOT / "result/tracking.csv",
    )
    parser.add_argument(
        "--amodal-head-checkpoint",
        type=Path,
        help="Run amodal-v10 from the same SAM3 pass using this checkpoint.",
    )
    parser.add_argument(
        "--amodal-cache-root",
        type=Path,
        help="Write one-pass amodal-v10 sequence NPZ files here.",
    )
    parser.add_argument("--amodal-amp", action="store_true")
    # Frozen after the eight-sequence VIS/RedNIR/NIR development check.
    parser.add_argument("--min-prototype-similarity", type=float, default=0.96)
    parser.add_argument("--min-upper-ratio", type=float, default=0.85)
    parser.add_argument("--min-local-contrast-ratio", type=float, default=0.25)
    parser.add_argument("--spatial-retention", type=float, default=0.35)
    parser.add_argument("--advantage-retention", type=float, default=0.10)
    parser.add_argument("--reference-containment-retention", type=float, default=0.10)
    return parser.parse_args()


def load_sequences(config):
    identifiers = read_sample_ids(config.data.sample_submission)
    required = {split_identifier(identifier)[0] for identifier in identifiers}
    initial_boxes = discover_init_bboxes(config.data.validation_root, required)
    sequences = load_validation_sequences(
        config.data.validation_root,
        config.data.sample_submission,
        initial_boxes,
    )
    return identifiers, initial_boxes, sequences


def assign_shards(sequences, count: int):
    shards = [[] for _ in range(count)]
    loads = [0] * count
    for sequence in sorted(sequences, key=lambda item: len(item.frames), reverse=True):
        index = min(range(count), key=loads.__getitem__)
        shards[index].append(sequence)
        loads[index] += len(sequence.frames)
    return shards, loads


def result_path(output_dir: Path, sequence_name: str) -> Path:
    return output_dir / "predictions" / f"{sequence_name.replace('/', '__')}.npy"


def diagnostic_path(output_dir: Path, sequence_name: str) -> Path:
    return output_dir / "diagnostics" / f"{sequence_name.replace('/', '__')}.json"


def valid_result(path: Path, frames: int) -> bool:
    try:
        return np.load(path, mmap_mode="r").shape == (frames, 4)
    except (OSError, ValueError):
        return False


def valid_amodal_result(path: Path, ids: tuple[str, ...]) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {"ids", "image_size", "baseline", "head", "mask_nonempty"}
            if not required.issubset(data.files):
                return False
            if tuple(str(value) for value in data["ids"].tolist()) != ids:
                return False
            count = len(ids)
            return bool(
                data["image_size"].shape == (2,)
                and data["baseline"].shape == (count, 4)
                and data["head"].shape == (count, 4)
                and data["mask_nonempty"].shape == (count,)
                and np.isfinite(data["baseline"]).all()
                and np.isfinite(data["head"]).all()
            )
    except (OSError, ValueError):
        return False


def hsi_sequence_available(frames) -> bool:
    try:
        for frame in frames:
            raw_hsi_path(frame)
    except HyperspectralError:
        return False
    return True


def atomic_save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npy.part")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    temporary.replace(path)


def atomic_save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def integer_box(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    value = clip_box(np.asarray(box, dtype=np.float64), width, height)
    rounded = np.rint(value).astype(int)
    rounded[0] = np.clip(rounded[0], 0, width - 1)
    rounded[1] = np.clip(rounded[1], 0, height - 1)
    rounded[2] = np.clip(rounded[2], 1, width - rounded[0])
    rounded[3] = np.clip(rounded[3], 1, height - rounded[1])
    return tuple(map(int, rounded))


def clone_tracker_branch(tracker):
    branch = copy.copy(tracker)
    branch.inference_state = copy.deepcopy(tracker.inference_state)
    branch.object_sizes = list(tracker.object_sizes)
    branch.tracking_times = list(tracker.tracking_times)
    return branch


def install_image_encoder_counter(tracker) -> None:
    predictor = tracker.predictor
    if hasattr(predictor, "_hotc_original_forward_image"):
        return
    predictor._hotc_original_forward_image = predictor.forward_image
    predictor._hotc_image_encoder_calls = 0

    def counted_forward_image(*args, **kwargs):
        predictor._hotc_image_encoder_calls += 1
        return predictor._hotc_original_forward_image(*args, **kwargs)

    predictor.forward_image = counted_forward_image


def current_cached_feature(tracker):
    cached = tracker.inference_state["cached_features"].get(tracker.frame_index)
    if cached is None or cached[1] is None:
        raise RuntimeError(
            f"SAM3 did not retain image features for frame {tracker.frame_index}"
        )
    return cached


def track_sequence(
    tracker,
    sequence,
    initial_box,
    model_config,
    args,
    amodal_runtime: StreamingAmodalV10,
):
    import torch

    frames = sequence.frames
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    sensor = sequence.name.split("-", 1)[0]
    first = _open_rgb(frames[0])
    width, height = first.size
    first_color = np.asarray(first)
    first_gray = cv2.cvtColor(first_color, cv2.COLOR_RGB2GRAY)
    predictions = [tuple(map(int, initial_box))]
    attempts: list[dict] = []
    bbox_events: list[dict] = []
    bbox_restarts: list[int] = []
    bbox_policy = StaticBBoxContainmentPolicyV3()
    bbox_gate = StaticBBoxContainmentGate(bbox_policy)
    bbox_gate.initialize(initial_box)
    current_frame_path = frames[0]
    original_add_to_drm = None
    hsi_tracker = tracker
    encoder_calls_before = int(
        getattr(tracker.predictor, "_hotc_image_encoder_calls", 0)
    )
    amodal_sequence = amodal_runtime.start_sequence(initial_box, (width, height))
    _release_sequence(tracker)
    try:
        with torch.inference_mode():
            initial_output = tracker.initialize(first, None, bbox=initial_box)
            initial_mask = np.asarray(initial_output["pred_mask"]) > 0
            previous_region = initial_mask
            previous_gray = first_gray
            previous_color = first_color
            gate = ReferenceHSIIdentityGate.fit(
                load_cube(frames[0], sensor),
                initial_mask,
                min_prototype_similarity=args.min_prototype_similarity,
                min_upper_ratio=args.min_upper_ratio,
                min_local_contrast_ratio=args.min_local_contrast_ratio,
                spatial_retention=args.spatial_retention,
                advantage_retention=args.advantage_retention,
                reference_containment_retention=(
                    args.reference_containment_retention
                ),
            )
            hsi_tracker = clone_tracker_branch(tracker)
            direct_initial_output = tracker.track(first, init=True)
            direct_initial_mask = np.asarray(direct_initial_output["pred_mask"]) > 0
            direct_initial_box = _box_from_mask(
                direct_initial_mask,
                tuple(map(int, initial_box)),
                width,
                height,
            )
            amodal_sequence.observe(
                tracker,
                np.asarray(direct_initial_box),
                bool(direct_initial_mask.any()),
            )
            original_add_to_drm = tracker.predictor.add_to_drm

            def gated_add_to_drm(*, inference_state, frame_idx, obj_id):
                _, video_masks = tracker.predictor._get_orig_video_res_output(
                    inference_state, tracker.predictor.curr_out["pred_masks"]
                )
                mask = (video_masks[0, 0] > 0).cpu().numpy()
                match = gate.evaluate(load_cube(current_frame_path, sensor), mask)
                attempts.append(
                    {
                        "frame_index": int(frame_idx),
                        **match.to_dict(),
                        "vetoed": not match.accepted,
                    }
                )
                if not match.accepted:
                    return frame_idx, inference_state["obj_ids"], video_masks
                return original_add_to_drm(
                    inference_state=inference_state, frame_idx=frame_idx, obj_id=obj_id
                )

            last_box = tuple(map(int, initial_box))
            direct_last_box = np.asarray(direct_initial_box, dtype=np.float32)
            for frame_index, frame_path in enumerate(frames[1:], start=1):
                current_frame_path = frame_path
                image = _open_rgb(frame_path)
                current_color = np.asarray(image)
                current_gray = cv2.cvtColor(current_color, cv2.COLOR_RGB2GRAY)
                bbox_gate.observe_global_pair(
                    previous_gray, current_gray, previous_region
                )
                direct_output = tracker.track(image)
                direct_mask = np.asarray(direct_output["pred_mask"]) > 0
                direct_box = _box_from_mask(
                    direct_mask,
                    tuple(map(int, direct_last_box)),
                    width,
                    height,
                )
                direct_last_box = np.asarray(direct_box, dtype=np.float32)
                amodal_sequence.observe(
                    tracker,
                    direct_last_box,
                    bool(direct_mask.any()),
                )
                shared_feature = current_cached_feature(tracker)
                hsi_frame_index = hsi_tracker.frame_index + 1
                hsi_tracker.inference_state["cached_features"][
                    hsi_frame_index
                ] = shared_feature
                tracker.predictor.add_to_drm = gated_add_to_drm
                try:
                    output = hsi_tracker.track(image)
                finally:
                    tracker.predictor.add_to_drm = original_add_to_drm
                native_mask = np.asarray(output["pred_mask"]) > 0
                native_box = _box_from_mask(
                    native_mask,
                    last_box,
                    width,
                    height,
                )
                current_box = native_box
                current_region = native_mask
                decision = bbox_gate.inspect(
                    native_box=native_box,
                    frame_index=frame_index,
                    previous_image=previous_color,
                    current_image=current_color,
                )
                if decision.reason != "bbox_passthrough" or decision.restart:
                    bbox_events.append(
                        {"frame_index": frame_index, **decision.to_dict()}
                    )
                if decision.restart:
                    if decision.anchor_box is None:
                        raise RuntimeError(
                            "bbox containment restart requested without anchor"
                        )
                    current_box = integer_box(decision.anchor_box, width, height)
                    _release_sequence(hsi_tracker)
                    original_forward_image = tracker.predictor.forward_image
                    tracker.predictor.forward_image = lambda _image: shared_feature[1]
                    try:
                        hsi_tracker.initialize(image, None, bbox=current_box)
                    finally:
                        tracker.predictor.forward_image = original_forward_image
                    current_region = box_mask(current_gray.shape, current_box)
                    bbox_restarts.append(frame_index)
                hsi_tracker.inference_state["cached_features"] = {
                    hsi_tracker.frame_index: shared_feature
                }
                last_box = current_box
                predictions.append(last_box)
                previous_region = current_region
                previous_gray = current_gray
                previous_color = current_color
                prune_history(
                    hsi_tracker,
                    current_frame=hsi_tracker.frame_index,
                    history_frames=model_config.history_frames,
                )
    finally:
        if original_add_to_drm is not None:
            tracker.predictor.add_to_drm = original_add_to_drm
        _release_sequence(tracker)
        if hsi_tracker is not tracker:
            _release_sequence(hsi_tracker)
    encoder_calls = int(
        getattr(tracker.predictor, "_hotc_image_encoder_calls", 0)
    ) - encoder_calls_before
    if encoder_calls != len(frames):
        raise RuntimeError(
            f"shared SAM3 image encoder ran {encoder_calls} times for {len(frames)} frames"
        )
    diagnostics = {
        "format": "dam4sam3-hsi-drm-gate-v3-kaggle75-sequence",
        "sequence": sequence.name,
        "frames": len(frames),
        "gate_calibration": gate.calibration_dict(),
        "drm_attempt_count": len(attempts),
        "drm_veto_count": sum(item["vetoed"] for item in attempts),
        "hsi_available": True,
        "drm_attempts": attempts,
        "static_bbox_containment_v3": True,
        "bbox_policy": bbox_policy.__dict__,
        "bbox_scene": bbox_gate.scene_diagnostics(),
        "bbox_restart_count": len(bbox_restarts),
        "bbox_restart_frames_zero_based": bbox_restarts,
        "bbox_events": bbox_events,
        "one_pass_amodal_v10": True,
        "sam3_image_encoder_calls": encoder_calls,
    }
    amodal_arrays = amodal_sequence.arrays(tuple(sequence.ids[: len(frames)]))
    return np.asarray(predictions, dtype=np.int32), diagnostics, amodal_arrays


def finalize(args, config, identifiers, sequences) -> int:
    predictions = {}
    diagnostics = []
    for sequence in sequences:
        path = result_path(args.output_dir, sequence.name)
        if not valid_result(path, len(sequence.frames)):
            raise RuntimeError(f"missing or invalid prediction cache: {path}")
        boxes = np.load(path).astype(int).tolist()
        predictions.update(zip(sequence.ids, boxes))
        diagnostics.append(json.loads(diagnostic_path(args.output_dir, sequence.name).read_text()))
    if len(predictions) != len(identifiers):
        raise RuntimeError(f"prediction count {len(predictions)} != {len(identifiers)}")
    args.submission.parent.mkdir(parents=True, exist_ok=True)
    summary = write_submission(config.data.sample_submission, args.submission, predictions)
    validated = validate_submission(args.submission, config.data.sample_submission)
    payload = {
        "format": "dam4sam3-hsi-drm-gate-v3-kaggle75",
        "gate_version": "v3",
        "static_bbox_containment_v3": True,
        "sequence_count": len(sequences),
        "frame_count": len(identifiers),
        "hsi_sequence_count": sum(bool(x.get("hsi_available")) for x in diagnostics),
        "hsi_fallback_sequence_count": sum(
            not bool(x.get("hsi_available")) for x in diagnostics
        ),
        "drm_attempt_count": sum(x["drm_attempt_count"] for x in diagnostics),
        "drm_veto_count": sum(x["drm_veto_count"] for x in diagnostics),
        "bbox_restart_count": sum(x.get("bbox_restart_count", 0) for x in diagnostics),
        "bbox_restart_sequence_count": sum(
            bool(x.get("bbox_restart_count", 0)) for x in diagnostics
        ),
        "submission": str(args.submission.resolve()),
        "summary": summary.__dict__,
        "validated": validated.__dict__,
        "frozen_gate": {
            "min_prototype_similarity": args.min_prototype_similarity,
            "min_upper_ratio": args.min_upper_ratio,
            "min_local_contrast_ratio": args.min_local_contrast_ratio,
            "spatial_retention": args.spatial_retention,
            "advantage_retention": args.advantage_retention,
            "reference_containment_retention": (
                args.reference_containment_retention
            ),
        },
    }
    atomic_save_json(args.output_dir / "manifest.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def main() -> int:
    import torch

    args = parse_args()
    if not args.finalize and (
        args.amodal_head_checkpoint is None or args.amodal_cache_root is None
    ):
        raise ValueError(
            "--amodal-head-checkpoint and --amodal-cache-root are required"
        )
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    config = load_config(args.config)
    if args.validation_root is not None or args.sample is not None:
        config = replace(
            config,
            data=replace(
                config.data,
                validation_root=(
                    args.validation_root.resolve()
                    if args.validation_root is not None
                    else config.data.validation_root
                ),
                sample_submission=(
                    args.sample.resolve()
                    if args.sample is not None
                    else config.data.sample_submission
                ),
            ),
        )
    identifiers, initial_boxes, sequences = load_sequences(config)
    if args.finalize:
        return finalize(args, config, identifiers, sequences)
    if args.sequence:
        requested = set(args.sequence)
        sequences = [sequence for sequence in sequences if sequence.name in requested]
        missing = requested.difference(sequence.name for sequence in sequences)
        if missing:
            raise ValueError(f"unknown sequences: {sorted(missing)}")
    if args.max_sequences is not None:
        sequences = sequences[: args.max_sequences]
    shards, loads = assign_shards(sequences, args.num_shards)
    assigned = shards[args.shard_index]
    model_config = config.model
    amodal_runtime = StreamingAmodalV10.from_checkpoint(
        args.amodal_head_checkpoint,
        device=args.device,
        amp=args.amodal_amp,
    )
    tracker = None
    records = []
    started = time.perf_counter()
    try:
        tracker = build_amodal_tracker(device=args.device)
        amodal_runtime.install_capture_hooks(tracker)
        install_image_encoder_counter(tracker)
        for index, sequence in enumerate(assigned, start=1):
            frames = min(
                len(sequence.frames),
                args.max_frames if args.max_frames is not None else len(sequence.frames),
            )
            path = result_path(args.output_dir, sequence.name)
            sequence_ids = tuple(sequence.ids[:frames])
            amodal_path = args.amodal_cache_root / f"{sequence.name}.npz"
            tracking_cached = valid_result(path, frames)
            amodal_cached = valid_amodal_result(amodal_path, sequence_ids)
            if tracking_cached and amodal_cached:
                records.append({"sequence": sequence.name, "frames": frames, "cached": True})
                print(f"[{index}/{len(assigned)}] {sequence.name}: cached", flush=True)
                continue
            if tracking_cached or amodal_path.exists():
                raise RuntimeError(
                    f"incomplete one-pass cache for {sequence.name}; remove output and rerun"
                )
            selected_frames = sequence.frames[:frames]
            if not hsi_sequence_available(selected_frames):
                raise RuntimeError(
                    f"raw HSI is required but unavailable for {sequence.name}"
                )
            sequence_started = time.perf_counter()
            boxes, diagnostics, amodal_arrays = track_sequence(
                tracker,
                sequence,
                initial_boxes[sequence.name],
                model_config,
                args,
                amodal_runtime,
            )
            atomic_save_array(path, boxes)
            atomic_save_json(diagnostic_path(args.output_dir, sequence.name), diagnostics)
            atomic_save_npz(amodal_path, amodal_arrays)
            elapsed = time.perf_counter() - sequence_started
            records.append(
                {
                    "sequence": sequence.name,
                    "frames": len(boxes),
                    "cached": False,
                    "seconds": elapsed,
                    "drm_attempts": diagnostics["drm_attempt_count"],
                    "drm_vetoes": diagnostics["drm_veto_count"],
                    "bbox_restarts": diagnostics["bbox_restart_count"],
                    "one_pass_amodal_v10": True,
                    "sam3_image_encoder_calls": diagnostics[
                        "sam3_image_encoder_calls"
                    ],
                }
            )
            print(
                f"[{index}/{len(assigned)}] {sequence.name}: {len(boxes)} frames, "
                f"{diagnostics['drm_veto_count']}/{diagnostics['drm_attempt_count']} veto, "
                f"bbox resets={diagnostics['bbox_restart_count']}, "
                f"{len(boxes) / elapsed:.2f} FPS",
                flush=True,
            )
    finally:
        if tracker is not None:
            _release_sequence(tracker)
        torch.cuda.empty_cache()
    payload = {
        "format": "dam4sam3-hsi-drm-gate-v3-kaggle75-shard",
        "gate_version": "v3",
        "static_bbox_containment_v3": True,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "assigned_frames": loads[args.shard_index],
        "elapsed_seconds": time.perf_counter() - started,
        "device": args.device,
        "one_pass_amodal_v10": True,
        "amodal_head_precision": "bf16" if amodal_runtime.amp else "float32",
        "records": records,
    }
    atomic_save_json(args.output_dir / f"shard-{args.shard_index}.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
