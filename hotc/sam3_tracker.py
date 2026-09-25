"""HOTC streaming tracker built on the pinned official Meta SAM 3 source.

This SAM 3 integration file is distributed under the SAM License in
``THIRD_PARTY_LICENSES/SAM_LICENSE``.
"""

from __future__ import annotations

import random
from pathlib import Path
from types import MethodType

import cv2
import numpy as np


def _is_visible(output) -> bool:
    if output is None:
        return False
    count = output.get("n_pixels_pos")
    if count is None:
        return True
    try:
        import torch

        if isinstance(count, torch.Tensor):
            return bool((count > 0).any().item())
    except ImportError:
        pass
    return bool(count > 0)


def _select_drm_frames(
    frame_index: int,
    conditioning: dict,
    maximum: int,
    *,
    keep_first: bool,
) -> tuple[dict, dict]:
    """Choose the immutable anchor and the nearest eligible DRM frames."""
    if maximum == 0 or not conditioning:
        return {}, dict(conditioning)

    chosen = {}
    if 0 in conditioning:
        chosen[0] = conditioning[0]
    elif keep_first:
        first = min(conditioning)
        chosen[first] = conditioning[first]

    previous = max(
        (index for index in conditioning if index < frame_index - 1),
        default=None,
    )
    if previous is not None:
        chosen[previous] = conditioning[previous]

    remaining = len(conditioning) if maximum == -1 else max(maximum - len(chosen), 0)
    nearest = sorted(
        (
            index
            for index in conditioning
            if index not in chosen and index != frame_index - 1
        ),
        key=lambda index: abs(index - frame_index),
    )[:remaining]
    chosen.update((index, conditioning[index]) for index in nearest)
    rejected = {
        index: output
        for index, output in conditioning.items()
        if index not in chosen
    }
    return chosen, rejected


def _prepare_dam_memory(
    self,
    frame_idx,
    is_init_cond_frame,
    current_vision_feats,
    current_vision_pos_embeds,
    feat_sizes,
    output_dict,
    num_frames,
    track_in_reverse=False,
    use_prev_mem_frame=True,
):
    """Assemble fixed-size DRM/RAM memory for the official SAM 3 encoder."""
    import torch

    batch = current_vision_feats[-1].size(1)
    channels = self.hidden_dim
    height, width = feat_sizes[-1]
    device = current_vision_feats[-1].device

    if self.num_maskmem == 0:
        return current_vision_feats[-1].permute(1, 2, 0).view(
            batch, channels, height, width
        )
    if is_init_cond_frame or not use_prev_mem_frame:
        values = current_vision_feats[-1] + self.no_mem_embed
        return values.permute(1, 2, 0).view(batch, channels, height, width)

    conditioning = output_dict["cond_frame_outputs"]
    if not conditioning:
        raise RuntimeError("SAM 3 tracking requires a conditioning frame")
    selected, unselected = _select_drm_frames(
        frame_idx,
        conditioning,
        self.max_cond_frames_in_attn,
        keep_first=self.keep_first_cond_frame,
    )

    memory_entries = [(0, output, True) for output in selected.values()]
    stride = 1 if self.training else self.memory_temporal_stride_for_eval
    slot_count = max(self.num_maskmem - len(memory_entries), 0)
    previous_index = frame_idx
    recent = []
    recent_indices = []

    for slot in range(slot_count):
        if slot == 0:
            previous_index = frame_idx - 1
            if previous_index in selected:
                output = None
            elif previous_index in conditioning:
                output = conditioning.get(previous_index)
            else:
                output = output_dict["non_cond_frame_outputs"].get(previous_index)
            if not _is_visible(output):
                while previous_index > 0:
                    previous_index -= 1
                    output = output_dict["non_cond_frame_outputs"].get(previous_index)
                    if _is_visible(output) and previous_index not in selected:
                        break
                else:
                    output = None
        elif previous_index >= 0:
            previous_index = ((previous_index - 1) // stride) * stride
            output = output_dict["non_cond_frame_outputs"].get(previous_index)
            if not _is_visible(output) or previous_index in selected:
                while previous_index > 0:
                    previous_index -= stride
                    output = output_dict["non_cond_frame_outputs"].get(previous_index)
                    if _is_visible(output) and previous_index not in selected:
                        break
                else:
                    output = None
        else:
            output = None
        recent.append(output)
        recent_indices.append(previous_index)

    memory_entries.extend(
        (index, output, False) for index, output in zip(recent_indices, recent)
    )
    memory_entries.sort(key=lambda item: item[0])
    ordered_entries = []
    for position, (index, output, is_drm) in enumerate(memory_entries):
        temporal_position = 0 if index == 0 else position
        ordered_entries.append((temporal_position, output, is_drm))

    memory_tokens = []
    memory_positions = []
    pointer_token_count = 0
    total_slots = len(ordered_entries)
    for temporal_position, output, is_drm in ordered_entries:
        if output is None:
            continue
        features = output["maskmem_features"].cuda(non_blocking=True)
        memory_tokens.append(features.flatten(2).permute(2, 0, 1))
        position = output["maskmem_pos_enc"][-1].cuda()
        position = position.flatten(2).permute(2, 0, 1)
        spatial_embedding = getattr(self, "cond_frame_spatial_embedding", None)
        if is_drm and spatial_embedding is not None:
            position = position + spatial_embedding
        position = position + self.maskmem_tpos_enc[
            total_slots - temporal_position - 1
        ]
        memory_positions.append(position)

    maximum_pointers = min(num_frames, self.max_obj_ptrs_in_encoder)
    if self.training:
        pointer_conditioning = selected
    else:
        pointer_conditioning = {
            index: output
            for index, output in selected.items()
            if (index >= frame_idx if track_in_reverse else index <= frame_idx)
        }
    pointer_entries = [
        (abs(frame_idx - index), output["obj_ptr"], True)
        for index, output in pointer_conditioning.items()
    ]
    for difference in range(1, maximum_pointers):
        index = frame_idx + difference if track_in_reverse else frame_idx - difference
        if index < 0 or index >= num_frames:
            break
        output = output_dict["non_cond_frame_outputs"].get(
            index, unselected.get(index)
        )
        if _is_visible(output):
            pointer_entries.append((difference, output["obj_ptr"], False))

    if pointer_entries:
        pointer_positions, pointers, is_drm_pointer = zip(*pointer_entries)
        pointer_tokens = torch.stack(pointers, dim=0)
        pointer_embedding = getattr(self, "cond_frame_obj_ptr_embedding", None)
        if pointer_embedding is not None:
            pointer_tokens = pointer_tokens + pointer_embedding * torch.tensor(
                is_drm_pointer, device=device
            )[..., None, None].float()
        encoded_positions = self._get_tpos_enc(
            pointer_positions,
            max_abs_pos=maximum_pointers,
            device=device,
        ).unsqueeze(1).expand(-1, batch, -1)
        if self.mem_dim < channels:
            pointer_tokens = pointer_tokens.reshape(
                -1, batch, channels // self.mem_dim, self.mem_dim
            )
            pointer_tokens = pointer_tokens.permute(0, 2, 1, 3).flatten(0, 1)
            encoded_positions = encoded_positions.repeat_interleave(
                channels // self.mem_dim, dim=0
            )
        memory_tokens.append(pointer_tokens)
        memory_positions.append(encoded_positions)
        pointer_token_count = pointer_tokens.shape[0]

    prompt = torch.cat(memory_tokens, dim=0)
    prompt_positions = torch.cat(memory_positions, dim=0)
    encoded = self.transformer.encoder(
        src=current_vision_feats,
        src_key_padding_mask=[None],
        src_pos=current_vision_pos_embeds,
        prompt=prompt,
        prompt_pos=prompt_positions,
        prompt_key_padding_mask=None,
        feat_sizes=feat_sizes,
        num_obj_ptr_tokens=pointer_token_count,
    )
    return encoded["memory"].permute(1, 2, 0).view(
        batch, channels, height, width
    )


def _mask_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, columns = np.nonzero(mask)
    if not len(columns):
        return 0, 0, 0, 0
    left = int(columns.min())
    top = int(rows.min())
    return left, top, int(columns.max()) - left, int(rows.max()) - top


def _box_iou(first, second) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    intersection_width = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    intersection_height = max(0, min(ay + ah, by + bh) - max(ay, by))
    intersection = intersection_width * intersection_height
    union = aw * ah + bw * bh - intersection
    return float(intersection / union) if union else 1.0


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8, copy=False), connectivity=8
    )
    if count <= 1:
        return np.zeros_like(mask, dtype=np.uint8)
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == component).astype(np.uint8)


def _install_drm_promotion(predictor) -> None:
    def add_to_drm(self, *, inference_state, frame_idx, obj_id):
        object_index = self._obj_id_to_idx(inference_state, obj_id)
        inference_state["mask_inputs_per_obj"][object_index].pop(frame_idx, None)
        inference_state["adds_in_drm_per_obj"][object_index].append(frame_idx)
        temporary = inference_state["temp_output_dict_per_obj"][object_index]
        temporary["cond_frame_outputs"][frame_idx] = self.curr_out
        consolidated = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=True,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        _, masks = self._get_orig_video_res_output(
            inference_state, consolidated["pred_masks_video_res"]
        )
        return frame_idx, inference_state["obj_ids"], masks

    predictor.add_to_drm = MethodType(add_to_drm, predictor)


class Sam3Tracker:
    """Single-object streaming DAM tracker using only official SAM 3 source."""

    def __init__(self, predictor):
        import torch

        self.predictor = predictor
        self.input_image_size = getattr(predictor, "image_size", 1008)
        self.img_mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)[:, None, None]
        self.img_std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)[:, None, None]
        self.tracking_times = []
        self._needs_preflight = False
        predictor._hotc_candidate_output = None
        predictor._prepare_memory_conditioned_features = MethodType(
            _prepare_dam_memory, predictor
        )
        predictor.multimask_min_pt_num = 0
        predictor.multimask_max_pt_num = 3
        predictor.memory_temporal_stride_for_eval = 5
        predictor.max_cond_frames_in_attn = 4
        _install_drm_promotion(predictor)
        self._candidate_hook = predictor.sam_mask_decoder.register_forward_hook(
            self._capture_candidates
        )

    def _capture_candidates(self, _module, _inputs, output) -> None:
        import torch

        if not isinstance(output, (tuple, list)) or len(output) < 4:
            raise RuntimeError("unexpected SAM 3 mask decoder output")
        masks, ious, _, presence = output[:4]
        if not all(isinstance(value, torch.Tensor) for value in (masks, ious, presence)):
            raise RuntimeError("SAM 3 mask decoder did not return tensors")
        self.predictor._hotc_candidate_output = (
            masks.detach().float().clone(),
            ious.detach().float().clone(),
            presence.detach().float().clone(),
        )

    def _prepare_image(self, image):
        import torch
        import torchvision.transforms.functional as functional

        device = self.inference_state.get(
            "storage_device", self.inference_state["device"]
        )
        values = torch.from_numpy(np.array(image.convert("RGB"))).to(device)
        values = values.permute(2, 0, 1).float() / 255.0
        values = functional.resize(
            values, (self.input_image_size, self.input_image_size)
        )
        return (values - self.img_mean.to(device)) / self.img_std.to(device)

    def initialize(self, image, init_mask, bbox=None):
        import torch

        if isinstance(init_mask, list):
            init_mask = init_mask[0]
        self.frame_index = 0
        self.object_sizes = []
        self.last_added = -1
        self.img_width, self.img_height = image.size
        self.inference_state = self.predictor.init_state(
            video_height=self.img_height,
            video_width=self.img_width,
            num_frames=1,
        )
        self.inference_state.setdefault("images", {})[0] = self._prepare_image(image)
        self.inference_state["num_frames"] = 1
        self.inference_state["adds_in_drm_per_obj"] = {}
        self._needs_preflight = True

        if init_mask is None:
            if bbox is None:
                raise ValueError("bbox or init_mask is required")
            x, y, width, height = bbox
            box = np.asarray(
                [[
                    x / self.img_width,
                    y / self.img_height,
                    (x + width) / self.img_width,
                    (y + height) / self.img_height,
                ]],
                dtype=np.float32,
            )
            _, _, _, masks = self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=0,
                obj_id=0,
                box=box,
            )
        else:
            _, _, _, masks = self.predictor.add_new_mask(
                inference_state=self.inference_state,
                frame_idx=0,
                obj_id=0,
                mask=torch.as_tensor(init_mask, dtype=torch.float32),
            )
        self.inference_state["adds_in_drm_per_obj"][0] = []
        return {"pred_mask": (masks[0, 0] > 0).cpu().numpy().astype(np.uint8)}

    def _preflight(self) -> None:
        inserted = []
        for object_index, frame_indices in self.inference_state[
            "adds_in_drm_per_obj"
        ].items():
            masks = self.inference_state["mask_inputs_per_obj"][object_index]
            points = self.inference_state["point_inputs_per_obj"][object_index]
            for frame_index in frame_indices:
                if frame_index not in masks and frame_index not in points:
                    masks[frame_index] = None
                    inserted.append((masks, frame_index))
        try:
            self.predictor.propagate_in_video_preflight(
                self.inference_state, run_mem_encoder=True
            )
        finally:
            for mapping, frame_index in inserted:
                if mapping.get(frame_index) is None:
                    mapping.pop(frame_index, None)

    def _candidate_masks(self):
        import torch

        candidate_output = self.predictor._hotc_candidate_output
        if candidate_output is None:
            return [], np.empty(0, dtype=np.float32)
        masks, ious, presence = candidate_output
        if not bool((presence > 0).all().item()):
            masks = torch.full_like(masks, -1024.0)
        masks = masks.squeeze()
        if masks.ndim == 2:
            masks = masks[None]
        resized = []
        for mask in masks:
            _, video_mask = self.predictor._get_orig_video_res_output(
                self.inference_state, mask[None, None]
            )
            resized.append(video_mask)
        return resized, ious.squeeze().cpu().numpy().copy()

    def track(self, image, init=False):
        self.inference_state["images"][self.frame_index if init else self.frame_index + 1] = (
            self._prepare_image(image)
        )
        if not init:
            self.frame_index += 1
        self.inference_state["num_frames"] += 1

        has_temporary = any(
            outputs[storage]
            for outputs in self.inference_state["temp_output_dict_per_obj"].values()
            for storage in ("cond_frame_outputs", "non_cond_frame_outputs")
        )
        if self._needs_preflight or has_temporary:
            self._preflight()

        result = None
        current = None
        for output in self.predictor.propagate_in_video(
            self.inference_state,
            start_frame_idx=self.frame_index,
            max_frame_num_to_track=0,
            reverse=False,
            tqdm_disable=True,
            propagate_preflight=False,
        ):
            out_frame_idx, _, _, video_masks, _ = output
            result = (video_masks[0, 0] > 0).cpu().numpy().astype(np.uint8)
            for storage in ("cond_frame_outputs", "non_cond_frame_outputs"):
                current = self.inference_state["output_dict"][storage].get(
                    out_frame_idx
                )
                if current is not None:
                    break

        if result is None or current is None:
            raise RuntimeError(f"SAM 3 produced no output for frame {self.frame_index}")
        current["n_pixels_pos"] = int(
            (current["pred_masks"][0, 0] > 0).detach().cpu().numpy().sum()
        )
        current["iou"] = 1.0
        self.predictor.curr_out = current

        candidates, candidate_ious = self._candidate_masks()
        selected_index = int(np.argmax(candidate_ious)) if candidate_ious.size else 0
        selected_iou = (
            float(candidate_ious[selected_index]) if candidate_ious.size else -np.inf
        )
        alternatives = [
            value for index, value in enumerate(candidates) if index != selected_index
        ]

        pixel_count = int(result.sum())
        self.object_sizes.append(pixel_count)
        if len(self.object_sizes) > 1 and pixel_count >= 1:
            recent = [value for value in self.object_sizes[-300:] if value >= 1][-10:]
            size_ratio = pixel_count / np.median(recent)
        else:
            size_ratio = -1.0

        interval_ready = self.last_added == -1 or self.frame_index - self.last_added > 5
        if (
            selected_iou > 0.8
            and 0.8 <= size_ratio <= 1.2
            and pixel_count >= 1
            and interval_ready
        ):
            binary_alternatives = [
                (value[0, 0] > 0).cpu().numpy().astype(np.uint8)
                for value in alternatives
            ]
            binary_alternatives = [
                np.logical_and(value, np.logical_not(result)).astype(np.uint8)
                for value in binary_alternatives
            ]
            binary_alternatives = [
                _largest_component(value)
                for value in binary_alternatives
                if int(value.sum()) >= 1
            ]
            expanded = [
                np.logical_or(value, result).astype(np.uint8)
                for value in binary_alternatives
            ]
            if expanded:
                chosen_box = _mask_box(result)
                overlaps = [_box_iou(chosen_box, _mask_box(value)) for value in expanded]
                if min(overlaps) <= 0.7:
                    self.last_added = self.frame_index
                    self.predictor.add_to_drm(
                        inference_state=self.inference_state,
                        frame_idx=self.frame_index,
                        obj_id=0,
                    )

        self._needs_preflight = False
        self.inference_state["images"].pop(self.frame_index, None)
        if init:
            self.inference_state["images"].pop(0, None)
        return {"pred_mask": result}


def build_tracker(source_root: Path, checkpoint: Path, device: str) -> Sam3Tracker:
    """Build the adapter from a pinned checkout of the official SAM 3 source."""
    import os
    import sys
    import torch

    source_root = source_root.resolve()
    checkpoint = checkpoint.resolve()
    if not (source_root / "sam3/model_builder.py").is_file():
        raise RuntimeError(f"missing official SAM 3 source: {source_root}")
    if not checkpoint.is_file():
        raise RuntimeError(f"missing SAM 3 checkpoint: {checkpoint}")

    torch.cuda.set_device(torch.device(device))
    random.seed(0)
    os.environ["PYTHONHASHSEED"] = "0"
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    source = str(source_root)
    loaded_sam3 = sys.modules.get("sam3")
    if loaded_sam3 is not None:
        loaded_file = getattr(loaded_sam3, "__file__", None)
        if loaded_file is None or source_root not in Path(loaded_file).resolve().parents:
            raise RuntimeError(
                "a different sam3 package is already loaded; start a clean process"
            )
    if source in sys.path:
        sys.path.remove(source)
    sys.path.insert(0, source)

    from sam3.model_builder import build_sam3_video_model

    model = build_sam3_video_model(
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
        device=device,
        apply_temporal_disambiguation=False,
    )
    predictor = model.tracker
    predictor.backbone = model.detector.backbone
    tracker = Sam3Tracker(predictor)
    tracker._amodal_checkpoint = str(checkpoint)
    tracker._amodal_device = device
    return tracker
