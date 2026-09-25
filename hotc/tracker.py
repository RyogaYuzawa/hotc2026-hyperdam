"""Small helpers around the pinned SAM3 tracker."""

from __future__ import annotations

import gc
from pathlib import Path


def _release_sequence(tracker) -> None:
    import torch

    state = getattr(tracker, "inference_state", None)
    if state is not None:
        try:
            tracker.predictor.reset_state(state)
        except (AttributeError, KeyError, RuntimeError):
            pass
        tracker.inference_state = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _box_from_mask(
    mask,
    fallback: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    import numpy as np

    points = np.argwhere(mask)
    if not len(points):
        return fallback
    y_min, x_min = points.min(axis=0).tolist()
    y_max, x_max = points.max(axis=0).tolist()
    x = max(0, min(int(x_min), width - 1))
    y = max(0, min(int(y_min), height - 1))
    box_width = max(1, min(int(x_max - x_min + 1), width - x))
    box_height = max(1, min(int(y_max - y_min + 1), height - y))
    return x, y, box_width, box_height


def _open_rgb(path: Path):
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB")


def _drop_old_mapping_entries(mapping, keep_frame) -> None:
    for frame_idx in list(mapping):
        if isinstance(frame_idx, int) and not keep_frame(frame_idx):
            mapping.pop(frame_idx, None)


def prune_history(tracker, current_frame: int, history_frames: int) -> None:
    """Keep the initial prompt anchor plus a rolling frame-history window."""
    state = tracker.inference_state
    if state is None:
        return
    cutoff = max(0, current_frame - history_frames + 1)
    anchor = state.get("first_ann_frame_idx")

    def keep(frame_idx: int) -> bool:
        return frame_idx == anchor or frame_idx >= cutoff

    for storage_key in ("cond_frame_outputs", "non_cond_frame_outputs"):
        _drop_old_mapping_entries(state["output_dict"][storage_key], keep)
        retained = {
            frame_idx
            for frame_idx in state["consolidated_frame_inds"][storage_key]
            if keep(frame_idx)
        }
        state["consolidated_frame_inds"][storage_key].intersection_update(retained)
        for outputs in state["output_dict_per_obj"].values():
            _drop_old_mapping_entries(outputs[storage_key], keep)
        for outputs in state["temp_output_dict_per_obj"].values():
            _drop_old_mapping_entries(outputs[storage_key], keep)

    for inputs_by_object in (
        state["point_inputs_per_obj"],
        state["mask_inputs_per_obj"],
    ):
        for inputs in inputs_by_object.values():
            _drop_old_mapping_entries(inputs, keep)
    for object_index, frame_indices in state["adds_in_drm_per_obj"].items():
        state["adds_in_drm_per_obj"][object_index] = [
            frame_idx for frame_idx in frame_indices if keep(frame_idx)
        ]
    _drop_old_mapping_entries(state["frames_already_tracked"], keep)
    _drop_old_mapping_entries(state["cached_features"], keep)
