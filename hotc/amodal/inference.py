from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .runtime import (
    current_spatial_decoder_features,
    current_temporal_head_inputs,
    install_mask_token_capture,
    install_spatial_decoder_capture,
)

from .model import SpatialTemporal3DConfig, SpatialTemporal3DAmodalRefiner


def align_mask_logits(
    logits: np.ndarray,
    box: np.ndarray,
    image_size: tuple[int, int],
    output_size: int,
    context: float,
    device: torch.device,
) -> np.ndarray:
    width, height = image_size
    normalized = np.asarray(box, dtype=np.float32).copy()
    normalized[[0, 2]] /= max(width, 1)
    normalized[[1, 3]] /= max(height, 1)
    center = normalized[:2] + 0.5 * normalized[2:]
    extent = np.maximum(normalized[2:] * context, 1e-4)
    axis = torch.linspace(-0.5, 0.5, output_size, device=device)
    grid_x = torch.as_tensor(center[0], device=device) + torch.as_tensor(
        extent[0], device=device
    ) * axis[None]
    grid_y = torch.as_tensor(center[1], device=device) + torch.as_tensor(
        extent[1], device=device
    ) * axis[:, None]
    grid = torch.stack(
        (
            2 * grid_x.expand(output_size, output_size) - 1,
            2 * grid_y.expand(output_size, output_size) - 1,
        ),
        dim=-1,
    )[None]
    values = torch.as_tensor(logits, dtype=torch.float32, device=device).reshape(
        1, 1, logits.shape[-2], logits.shape[-1]
    )
    return F.grid_sample(
        values,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0, 0].cpu().numpy()


class StreamingAmodalV10:
    def __init__(
        self,
        model: SpatialTemporal3DAmodalRefiner,
        *,
        threshold: float,
        device: torch.device,
        amp: bool,
    ) -> None:
        self.model = model
        self.threshold = float(threshold)
        self.device = device
        self.amp = bool(amp)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Path,
        *,
        device: str,
        amp: bool,
    ) -> "StreamingAmodalV10":
        torch_device = torch.device(device)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = SpatialTemporal3DAmodalRefiner(
            SpatialTemporal3DConfig(**payload["model_config"])
        )
        model.load_state_dict(payload["model"])
        model.to(torch_device).eval()
        return cls(
            model,
            threshold=float(payload.get("gate_threshold", 0.5)),
            device=torch_device,
            amp=amp,
        )

    @staticmethod
    def install_capture_hooks(tracker) -> None:
        install_mask_token_capture(tracker)
        install_spatial_decoder_capture(tracker)

    def start_sequence(
        self,
        initial_box: tuple[int, int, int, int],
        image_size: tuple[int, int],
    ) -> "StreamingAmodalSequence":
        return StreamingAmodalSequence(self, initial_box, image_size)


class StreamingAmodalSequence:
    def __init__(
        self,
        runtime: StreamingAmodalV10,
        initial_box: tuple[int, int, int, int],
        image_size: tuple[int, int],
    ) -> None:
        self.runtime = runtime
        self.initial_box = np.asarray(initial_box, dtype=np.float32)
        self.image_size = image_size
        self.detector_history: list[np.ndarray] = []
        self.feature_history: list[np.ndarray] = []
        self.logit_history: list[np.ndarray] = []
        self.query_history: list[np.ndarray] = []
        self.baseline: list[np.ndarray] = []
        self.head: list[np.ndarray] = []
        self.mask_nonempty: list[bool] = []

    @torch.inference_mode()
    def observe(self, tracker, modal_box: np.ndarray, mask_nonempty: bool) -> np.ndarray:
        model = self.runtime.model
        config = model.config
        device = self.runtime.device
        frame_index = len(self.baseline)
        modal_box = np.asarray(modal_box, dtype=np.float32)

        obj_ptr, _mask_token, raw_logits, _presence = current_temporal_head_inputs(
            tracker
        )
        aligned_logits = align_mask_logits(
            raw_logits,
            modal_box,
            self.image_size,
            config.spatial_size,
            config.spatial_context,
            device,
        ).astype(np.float16)
        spatial = current_spatial_decoder_features(
            tracker,
            modal_box,
            self.image_size,
            output_size=config.spatial_size,
            context=config.spatial_context,
        )
        self.detector_history.append(modal_box)
        self.feature_history.append(spatial)
        self.logit_history.append(aligned_logits)
        self.query_history.append(np.asarray(obj_ptr, dtype=np.float32))
        self.mask_nonempty.append(bool(mask_nonempty))

        start = max(0, frame_index - config.history_frames + 1)
        selected = list(range(start, frame_index + 1))
        padding = config.history_frames - len(selected)
        indices = [0] * padding + selected
        valid = np.asarray([False] * padding + [True] * len(selected), dtype=np.bool_)
        size = np.asarray([*self.image_size, *self.image_size], dtype=np.float32)
        detector = np.stack([self.detector_history[index] for index in indices]) / size
        features = np.stack([self.feature_history[index] for index in indices]).astype(
            np.float32
        )
        logits = np.stack([self.logit_history[index] for index in indices]).astype(
            np.float32
        )
        queries = np.stack([self.query_history[index] for index in indices]).astype(
            np.float32
        )
        features[~valid] = 0.0
        logits[~valid] = 0.0

        inputs = {
            "detector_boxes": torch.from_numpy(detector[None]).to(device),
            "decoder_features": torch.from_numpy(features[None]).to(device),
            "mask_logits": torch.from_numpy(logits[None, :, None]).to(device),
            "detector_queries": torch.from_numpy(queries[None]).to(device),
            "history_valid": torch.from_numpy(valid[None]).to(device),
        }
        if config.use_first_frame_anchor:
            inputs.update(
                {
                    "anchor_decoder_feature": torch.from_numpy(
                        self.feature_history[0][None].astype(np.float32)
                    ).to(device),
                    "anchor_mask_logit": torch.from_numpy(
                        self.logit_history[0][None, None].astype(np.float32)
                    ).to(device),
                    "anchor_query": torch.from_numpy(
                        self.query_history[0][None].astype(np.float32)
                    ).to(device),
                    "anchor_box": torch.from_numpy((self.initial_box / size)[None]).to(
                        device
                    ),
                }
            )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=self.runtime.amp and device.type == "cuda",
        ):
            prediction = model(**inputs, gate_threshold=self.runtime.threshold)
        proposed = prediction["boxes"][0].float().cpu().numpy() * size
        emitted_baseline = modal_box.copy()
        emitted_head = proposed if mask_nonempty else emitted_baseline.copy()
        if frame_index == 0:
            emitted_baseline = self.initial_box.copy()
            emitted_head = self.initial_box.copy()
        self.baseline.append(emitted_baseline)
        self.head.append(emitted_head)
        return emitted_head

    def arrays(self, ids: tuple[str, ...]) -> dict[str, np.ndarray]:
        if len(ids) != len(self.baseline):
            raise ValueError(
                f"amodal frame count {len(self.baseline)} does not match IDs {len(ids)}"
            )
        return {
            "ids": np.asarray(ids),
            "image_size": np.asarray(self.image_size, dtype=np.int64),
            "baseline": np.stack(self.baseline).astype(np.float32),
            "head": np.stack(self.head).astype(np.float32),
            "mask_nonempty": np.asarray(self.mask_nonempty, dtype=np.bool_),
        }
