"""Bounding-box decoding for amodal-v10 inference."""

from __future__ import annotations

import torch


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x, y, width, height = boxes.unbind(-1)
    return torch.stack((x, y, x + width, y + height), dim=-1)


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack((x1, y1, x2 - x1, y2 - y1), dim=-1)


def decode_outward_expansion(
    reference_xywh: torch.Tensor,
    expansion: torch.Tensor,
    *,
    clamp: bool = True,
    maximum: float = 4.0,
) -> torch.Tensor:
    """Expand a box without allowing any edge to move inward."""
    reference = xywh_to_xyxy(reference_xywh)
    width = reference_xywh[..., 2].clamp_min(1e-6)
    height = reference_xywh[..., 3].clamp_min(1e-6)
    left, top, right, bottom = expansion.clamp(0.0, maximum).unbind(-1)
    decoded = torch.stack(
        (
            reference[..., 0] - left * width,
            reference[..., 1] - top * height,
            reference[..., 2] + right * width,
            reference[..., 3] + bottom * height,
        ),
        dim=-1,
    )
    if clamp:
        decoded = decoded.clamp(0.0, 1.0)
    return xyxy_to_xywh(decoded)
