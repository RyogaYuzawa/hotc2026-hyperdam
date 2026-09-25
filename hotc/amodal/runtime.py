"""SAM3 feature capture used by the released amodal-v10 runtime."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAM3_ROOT = PROJECT_ROOT / "external/sam3"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "weights/sam3.pt"


class RuntimeError_(RuntimeError):
    pass


def build_tracker(checkpoint: str | Path = DEFAULT_CHECKPOINT, device: str = "cuda:0"):
    """Build the HOTC adapter on the pinned official SAM 3 source."""
    import torch

    from hotc.sam3_tracker import build_tracker as build_official_tracker

    checkpoint = Path(checkpoint).resolve()
    required = (SAM3_ROOT / "sam3/model_builder.py", checkpoint)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError_("missing official SAM 3 runtime files: " + ", ".join(missing))
    if not torch.cuda.is_available():
        raise RuntimeError_("CUDA is required for SAM 3 feature extraction")
    return build_official_tracker(SAM3_ROOT, checkpoint, device)


def current_amodal_head_inputs(tracker) -> tuple[object, object, float]:
    """Read the current object pointer, mask crop, and presence score."""
    import numpy as np
    import torch
    from torch.nn import functional as F

    current = getattr(tracker.predictor, "curr_out", None)
    if not isinstance(current, dict):
        raise RuntimeError_("DAM4SAM3 predictor has no current output")
    pointer = current.get("obj_ptr")
    logits = current.get("pred_masks")
    object_score = current.get("object_score_logits")
    if not isinstance(pointer, torch.Tensor) or not isinstance(logits, torch.Tensor):
        raise RuntimeError_("DAM4SAM3 current output lacks obj_ptr or pred_masks")
    token = pointer.detach().float().reshape(-1)
    if token.numel() < 256:
        token = F.pad(token, (0, 256 - token.numel()))
    token_array = token[:256].cpu().numpy().astype(np.float32, copy=False)

    mask = logits.detach().float().reshape(-1, *logits.shape[-2:])[0]
    foreground = torch.nonzero(mask > 0, as_tuple=False)
    if foreground.numel():
        top_left = foreground.min(dim=0).values
        bottom_right = foreground.max(dim=0).values + 1
        padding_y = max(1, int((bottom_right[0] - top_left[0]) * 0.1))
        padding_x = max(1, int((bottom_right[1] - top_left[1]) * 0.1))
        y1 = max(0, int(top_left[0]) - padding_y)
        x1 = max(0, int(top_left[1]) - padding_x)
        y2 = min(mask.shape[0], int(bottom_right[0]) + padding_y)
        x2 = min(mask.shape[1], int(bottom_right[1]) + padding_x)
        mask = mask[y1:y2, x1:x2]
    grid = F.interpolate(
        mask[None, None], size=(32, 32), mode="bilinear", align_corners=False
    )[0, 0]
    grid_array = grid.cpu().numpy().astype(np.float32, copy=False)
    presence = 1.0
    if isinstance(object_score, torch.Tensor):
        presence = float(torch.sigmoid(object_score.detach().float()).max().cpu())
    return token_array, grid_array, presence


def install_mask_token_capture(tracker) -> None:
    """Capture the selected frozen SAM3 mask token and raw mask logits."""
    import torch

    if getattr(tracker, "_amodal_mask_token_hooks", None):
        return

    predictor = tracker.predictor
    modules = []
    for name in ("sam_mask_decoder", "interactive_sam_mask_decoder"):
        module = getattr(predictor, name, None)
        if module is not None and module not in modules:
            modules.append(module)
    if not modules:
        raise RuntimeError_("SAM3 predictor exposes no mask decoder to capture")

    def capture(_module, _inputs, output):
        mask_tokens = ious = masks = None
        if isinstance(output, dict):
            mask_tokens = output.get("sam_tokens_out")
            ious = output.get("iou_pred")
            masks = output.get("masks")
        elif isinstance(output, (tuple, list)) and len(output) >= 3:
            masks = output[0]
            ious = output[1]
            mask_tokens = output[2]
        if not isinstance(mask_tokens, torch.Tensor) or not mask_tokens.numel():
            return
        values = mask_tokens.detach().float()
        while values.ndim > 3:
            values = values[:, 0]
            if isinstance(ious, torch.Tensor) and ious.ndim > 2:
                ious = ious[:, 0]
            if isinstance(masks, torch.Tensor) and masks.ndim > 4:
                masks = masks[:, 0]
        best = None
        if values.ndim == 3:
            if values.shape[1] == 1:
                selected = values[:, 0]
            elif isinstance(ious, torch.Tensor):
                scores = ious.detach().float().reshape(values.shape[0], -1)
                best = scores.argmax(dim=-1).clamp_max(values.shape[1] - 1)
                selected = values[torch.arange(len(values), device=values.device), best]
            else:
                selected = values[:, 0]
        else:
            selected = values.reshape(-1, values.shape[-1])
        tracker._amodal_mask_token = selected[0].cpu()
        if isinstance(masks, torch.Tensor) and masks.numel():
            mask_values = masks.detach().float()
            if mask_values.ndim == 3:
                mask_values = mask_values[:, None]
            if mask_values.ndim != 4:
                raise RuntimeError_(
                    f"unexpected SAM3 decoder mask shape {tuple(mask_values.shape)}"
                )
            if best is None:
                selected_mask = mask_values[:, 0]
            else:
                selected_mask = mask_values[
                    torch.arange(len(mask_values), device=mask_values.device), best
                ]
            tracker._amodal_raw_mask_logits = selected_mask[0].cpu()

    tracker._amodal_mask_token_hooks = [
        module.register_forward_hook(capture) for module in modules
    ]


def install_spatial_decoder_capture(tracker) -> None:
    """Capture the frozen SAM3 spatial decoder embedding."""
    import torch

    if getattr(tracker, "_amodal_spatial_decoder_hooks", None):
        return

    modules = []
    for name in ("sam_mask_decoder", "interactive_sam_mask_decoder"):
        decoder = getattr(tracker.predictor, name, None)
        upscaling = getattr(decoder, "output_upscaling", None)
        if upscaling is None or not len(upscaling):
            continue
        module = upscaling[-1]
        if module not in modules:
            modules.append(module)
    if not modules:
        raise RuntimeError_("SAM3 predictor exposes no spatial decoder embedding")

    def capture(_module, _inputs, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise RuntimeError_(
                "unexpected SAM3 spatial decoder feature shape "
                f"{getattr(output, 'shape', None)}"
            )
        tracker._amodal_spatial_decoder_feature = output.detach()

    tracker._amodal_spatial_decoder_hooks = [
        module.register_forward_hook(capture) for module in modules
    ]


def current_spatial_decoder_features(
    tracker,
    box,
    image_size: tuple[int, int],
    *,
    output_size: int = 24,
    context: float = 2.0,
):
    """Return a bbox-aligned CxHxW decoder ROI for temporal memory."""
    import numpy as np
    import torch
    from torch.nn import functional as F

    if output_size < 8 or context <= 0:
        raise ValueError("output_size must be >=8 and context must be positive")
    feature = getattr(tracker, "_amodal_spatial_decoder_feature", None)
    if not isinstance(feature, torch.Tensor) or feature.ndim != 4:
        raise RuntimeError_(
            "spatial decoder feature was not captured; call "
            "install_spatial_decoder_capture before initialize"
        )
    values = feature[:1].detach().float()
    width, height = image_size
    normalized = np.asarray(box, dtype=np.float32).copy()
    normalized[[0, 2]] /= max(width, 1)
    normalized[[1, 3]] /= max(height, 1)
    center = normalized[:2] + 0.5 * normalized[2:]
    extent = np.maximum(normalized[2:] * context, 1e-4)
    axis = torch.linspace(-0.5, 0.5, output_size, device=values.device)
    grid_x = torch.as_tensor(center[0], device=values.device) + torch.as_tensor(
        extent[0], device=values.device
    ) * axis[None, :]
    grid_y = torch.as_tensor(center[1], device=values.device) + torch.as_tensor(
        extent[1], device=values.device
    ) * axis[:, None]
    grid = torch.stack(
        (
            2.0 * grid_x.expand(output_size, output_size) - 1.0,
            2.0 * grid_y.expand(output_size, output_size) - 1.0,
        ),
        dim=-1,
    )[None]
    aligned = F.grid_sample(
        values, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )[0]
    return aligned.to(torch.float16).cpu().numpy()


def current_temporal_head_inputs(tracker) -> tuple[object, object, object, float]:
    """Return the object pointer, raw token/logits, and presence score."""
    import numpy as np
    import torch

    obj_ptr, _, presence = current_amodal_head_inputs(tracker)
    raw_token = getattr(tracker, "_amodal_mask_token", None)
    if not isinstance(raw_token, torch.Tensor):
        raise RuntimeError_(
            "mask token was not captured; call install_mask_token_capture before initialize"
        )
    token = raw_token.detach().float().reshape(-1)
    if token.numel() < 256:
        token = torch.nn.functional.pad(token, (0, 256 - token.numel()))
    token_array = token[:256].cpu().numpy().astype(np.float32, copy=False)
    raw_mask = getattr(tracker, "_amodal_raw_mask_logits", None)
    if not isinstance(raw_mask, torch.Tensor):
        raise RuntimeError_("raw mask logits were not captured from the mask decoder")
    mask_array = raw_mask.detach().float().cpu().numpy().astype(np.float32, copy=False)
    return obj_ptr, token_array, mask_array, presence
