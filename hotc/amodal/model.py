"""Spatiotemporal amodal-v10 model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .boxes import decode_outward_expansion


LEGACY_EXPANSION = "legacy_softplus_probability_scaled"
BOUNDED_RELATIVE_EXPANSION = "bounded_relative_sigmoid"
LOG_RELATIVE_EXPANSION = "log_relative_sigmoid_v6"
ANCHOR_FUSION_CROSS_ATTENTION = "cross_attention_v3"
ANCHOR_FUSION_EARLY_VOLUME = "early_volume_v4"


class SpatialTemporalBlock(nn.Module):
    """Separable 3-D residual block that preserves T, H, and W."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        groups = min(32, width)
        while width % groups:
            groups -= 1
        self.spatial = nn.Conv3d(width, width, (1, 3, 3), padding=(0, 1, 1))
        self.spatial_norm = nn.GroupNorm(groups, width)
        # A full 3x3x3 mixer is intentionally used here: capacity is preferred
        # over collapsing the decoder map into a cheap global frame vector.
        self.temporal = nn.Conv3d(width, width, 3, padding=1)
        self.temporal_norm = nn.GroupNorm(groups, width)
        self.dropout = nn.Dropout3d(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.spatial_norm(self.spatial(value)))
        hidden = self.temporal(hidden)
        hidden = self.dropout(F.gelu(self.temporal_norm(hidden)))
        return value + hidden


class AnchorSpatialBlock(nn.Module):
    """2-D residual block for the immutable first-frame spatial anchor."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        groups = min(32, width)
        while width % groups:
            groups -= 1
        self.conv1 = nn.Conv2d(width, width, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, width)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, width)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.norm1(self.conv1(value)))
        hidden = self.dropout(F.gelu(self.norm2(self.conv2(hidden))))
        return value + hidden


@dataclass(frozen=True)
class SpatialTemporal3DConfig:
    decoder_channels: int = 32
    history_frames: int = 6
    spatial_size: int = 24
    spatial_context: float = 2.0
    width: int = 192
    blocks: int = 6
    attention_heads: int = 8
    query_dim: int = 256
    geometry_dim: int = 14
    dropout: float = 0.1
    gate_power: float = 1.0
    max_expansion: float = 4.0
    expansion_parameterization: str = LEGACY_EXPANSION
    use_first_frame_anchor: bool = False
    anchor_fusion_mode: str = ANCHOR_FUSION_CROSS_ATTENTION
    reference_mode: str = "segment"
    per_side_hard_gate: bool = True
    predict_measurement_reliability: bool = False
    predict_correction_utility: bool = False
    correction_utility_threshold: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)


class SpatialTemporal3DAmodalRefiner(nn.Module):
    """External-memory 3-D decoder-feature head with four spatial side queries."""

    checkpoint_format = "dam4sam3-spatiotemporal3d-amodal-v1"
    bounded_checkpoint_format = "dam4sam3-spatiotemporal3d-amodal-bounded-ratio-v2"
    anchor_checkpoint_format = (
        "dam4sam3-spatiotemporal3d-amodal-bounded-ratio-first-frame-anchor-v3"
    )
    early_anchor_checkpoint_format = (
        "dam4sam3-spatiotemporal3d-amodal-bounded-ratio-early-anchor-v4"
    )
    log_reliability_checkpoint_format = (
        "dam4sam3-spatiotemporal3d-amodal-log-relative-reliability-v6"
    )
    utility_checkpoint_format = (
        "dam4sam3-spatiotemporal3d-amodal-log-relative-reliability-utility-v7"
    )

    def __init__(self, config: SpatialTemporal3DConfig = SpatialTemporal3DConfig()) -> None:
        super().__init__()
        if config.history_frames < 1 or config.spatial_size < 8:
            raise ValueError("invalid temporal/spatial dimensions")
        if config.width % config.attention_heads:
            raise ValueError("width must be divisible by attention_heads")
        if config.reference_mode != "segment" or not config.per_side_hard_gate:
            raise ValueError("3-D model requires segment anchor and per-side hard gate")
        if config.expansion_parameterization not in {
            LEGACY_EXPANSION,
            BOUNDED_RELATIVE_EXPANSION,
            LOG_RELATIVE_EXPANSION,
        }:
            raise ValueError(
                "unsupported expansion parameterization: "
                f"{config.expansion_parameterization}"
            )
        if config.max_expansion <= 0.0:
            raise ValueError("max_expansion must be positive")
        if not 0.0 <= config.correction_utility_threshold <= 1.0:
            raise ValueError("correction utility threshold must be in [0,1]")
        if (
            config.expansion_parameterization == BOUNDED_RELATIVE_EXPANSION
            and config.max_expansion > 1.0
        ):
            raise ValueError("bounded relative expansion must be in [0,1]")
        if (
            config.use_first_frame_anchor
            and config.expansion_parameterization
            not in {BOUNDED_RELATIVE_EXPANSION, LOG_RELATIVE_EXPANSION}
        ):
            raise ValueError(
                "first-frame anchor requires bounded or log-relative expansion"
            )
        if config.anchor_fusion_mode not in {
            ANCHOR_FUSION_CROSS_ATTENTION,
            ANCHOR_FUSION_EARLY_VOLUME,
        }:
            raise ValueError(f"unsupported anchor fusion: {config.anchor_fusion_mode}")
        if not config.use_first_frame_anchor and (
            config.anchor_fusion_mode != ANCHOR_FUSION_CROSS_ATTENTION
        ):
            raise ValueError("anchor fusion mode requires first-frame anchor inputs")
        self.config = config
        input_channels = config.decoder_channels + 3  # decoder, logit, x/y coordinates
        self.stem = nn.Sequential(
            nn.Conv3d(input_channels, config.width, 3, padding=1),
            nn.GroupNorm(min(32, config.width), config.width),
            nn.GELU(),
        )
        self.geometry_encoder = nn.Sequential(
            nn.LayerNorm(config.geometry_dim),
            nn.Linear(config.geometry_dim, config.width),
            nn.GELU(),
            nn.Linear(config.width, config.width),
        )
        self.query_encoder = nn.Sequential(
            nn.LayerNorm(config.query_dim),
            nn.Linear(config.query_dim, config.width),
            nn.GELU(),
            nn.Linear(config.width, config.width),
        )
        self.time_embedding = nn.Parameter(
            torch.zeros(1, config.width, config.history_frames, 1, 1)
        )
        self.blocks = nn.Sequential(
            *[SpatialTemporalBlock(config.width, config.dropout) for _ in range(config.blocks)]
        )
        self.final_norm = nn.GroupNorm(min(32, config.width), config.width)
        self.position = nn.Parameter(
            torch.zeros(1, config.width, config.spatial_size, config.spatial_size)
        )
        if (
            config.use_first_frame_anchor
            and config.anchor_fusion_mode == ANCHOR_FUSION_CROSS_ATTENTION
        ):
            anchor_groups = min(32, config.width)
            while config.width % anchor_groups:
                anchor_groups -= 1
            self.anchor_stem = nn.Sequential(
                nn.Conv2d(input_channels, config.width, 3, padding=1),
                nn.GroupNorm(anchor_groups, config.width),
                nn.GELU(),
                AnchorSpatialBlock(config.width, config.dropout),
                AnchorSpatialBlock(config.width, config.dropout),
            )
            self.anchor_geometry_encoder = nn.Sequential(
                nn.LayerNorm(12),
                nn.Linear(12, config.width),
                nn.GELU(),
                nn.Linear(config.width, config.width),
            )
            self.anchor_cross_attention = nn.MultiheadAttention(
                config.width,
                config.attention_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            self.anchor_cross_norm = nn.LayerNorm(config.width)
            # Exact residual identity at warm start. The scalar learns first;
            # the anchor branch then receives gradients without perturbing a
            # pretrained temporal body on step zero.
            self.anchor_residual_gate = nn.Parameter(torch.zeros(()))
        elif (
            config.use_first_frame_anchor
            and config.anchor_fusion_mode == ANCHOR_FUSION_EARLY_VOLUME
        ):
            # The immutable frame-zero feature is aligned to the current box,
            # repeated beside every valid history slice, and fused before all
            # residual 3-D blocks.  A single zero-initialized projection keeps
            # a warm-start checkpoint unchanged while receiving a direct
            # gradient on the first step (unlike a zero scalar residual gate).
            anchor_volume_channels = 2 * (config.decoder_channels + 1)
            self.anchor_volume_projection = nn.Conv3d(
                anchor_volume_channels,
                config.width,
                kernel_size=3,
                padding=1,
                bias=False,
            )
            nn.init.zeros_(self.anchor_volume_projection.weight)
            anchor_condition_dim = config.query_dim + 12
            self.anchor_condition_projection = nn.Sequential(
                nn.LayerNorm(anchor_condition_dim),
                nn.Linear(anchor_condition_dim, config.width),
                nn.GELU(),
                nn.Linear(config.width, config.width),
            )
            nn.init.zeros_(self.anchor_condition_projection[-1].weight)
            nn.init.zeros_(self.anchor_condition_projection[-1].bias)
        self.side_queries = nn.Parameter(torch.randn(1, 4, config.width) * 0.02)
        self.side_attention = nn.MultiheadAttention(
            config.width,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.side_trunk = nn.Sequential(
            nn.Linear(config.width * 3, config.width * 2),
            nn.LayerNorm(config.width * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.width * 2, config.width),
            nn.LayerNorm(config.width),
            nn.GELU(),
        )
        self.side_head = nn.Linear(config.width, 1)
        self.expansion_head = nn.Linear(config.width, 1)
        if config.predict_measurement_reliability:
            self.measurement_reliability_head = nn.Linear(config.width, 1)
            # Most frozen tracker observations are usable; initialize the
            # auxiliary router accordingly without changing bbox inference.
            nn.init.constant_(self.measurement_reliability_head.bias, math.log(4.0))
        if config.predict_correction_utility:
            utility_input = config.width * 4 + 8
            self.correction_utility_head = nn.Sequential(
                nn.LayerNorm(utility_input),
                nn.Linear(utility_input, config.width),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.width, 1),
            )
            # A conservative initial router preserves most baseline boxes.
            nn.init.constant_(self.correction_utility_head[-1].bias, -1.0)
        nn.init.constant_(self.side_head.bias, -4.0)
        if config.expansion_parameterization == BOUNDED_RELATIVE_EXPANSION:
            # Start near the observed hidden-edge median without making the
            # initially inactive side classifier alter any output boxes.
            nn.init.constant_(self.expansion_head.bias, -2.197224577)  # logit(0.1)
        elif config.expansion_parameterization == LOG_RELATIVE_EXPANSION:
            initial = math.log1p(0.1) / math.log1p(config.max_expansion)
            nn.init.constant_(
                self.expansion_head.bias, math.log(initial / (1.0 - initial))
            )
        else:
            nn.init.constant_(self.expansion_head.bias, -4.0)

    @property
    def active_checkpoint_format(self) -> str:
        if self.config.predict_correction_utility:
            return self.utility_checkpoint_format
        if self.config.expansion_parameterization == LOG_RELATIVE_EXPANSION:
            return self.log_reliability_checkpoint_format
        if self.config.use_first_frame_anchor:
            if self.config.expansion_parameterization != BOUNDED_RELATIVE_EXPANSION:
                raise ValueError("first-frame anchor requires bounded-ratio expansion")
            if self.config.anchor_fusion_mode == ANCHOR_FUSION_EARLY_VOLUME:
                return self.early_anchor_checkpoint_format
            return self.anchor_checkpoint_format
        if self.config.expansion_parameterization == BOUNDED_RELATIVE_EXPANSION:
            return self.bounded_checkpoint_format
        return self.checkpoint_format

    def _geometry(
        self, boxes: torch.Tensor, history_valid: torch.Tensor
    ) -> torch.Tensor:
        current = boxes[:, -1:, :]
        delta = boxes - current
        centers = boxes[..., :2] + 0.5 * boxes[..., 2:]
        current_center = centers[:, -1:, :]
        center_delta = (centers - current_center) / current[..., 2:].clamp_min(1e-5)
        log_size_ratio = torch.log(
            boxes[..., 2:].clamp_min(1e-5) / current[..., 2:].clamp_min(1e-5)
        )
        age = torch.linspace(
            -1.0, 0.0, boxes.shape[1], device=boxes.device, dtype=boxes.dtype
        )[None, :, None].expand(boxes.shape[0], -1, -1)
        return torch.cat(
            (
                boxes,
                delta,
                center_delta,
                log_size_ratio,
                age,
                history_valid[..., None].float(),
            ),
            dim=-1,
        )

    def _align_to_current(
        self, values: torch.Tensor, boxes: torch.Tensor
    ) -> torch.Tensor:
        """Warp independently cropped history ROIs into current-box coordinates."""
        batch, frames, channels, height, width = values.shape
        current = boxes[:, -1:, :]
        past_center = boxes[..., :2] + 0.5 * boxes[..., 2:]
        current_center = current[..., :2] + 0.5 * current[..., 2:]
        axis_x = torch.linspace(-1.0, 1.0, width, device=values.device, dtype=values.dtype)
        axis_y = torch.linspace(-1.0, 1.0, height, device=values.device, dtype=values.dtype)
        image_x = current_center[..., 0, None] + (
            axis_x[None, None] * self.config.spatial_context * current[..., 2, None] / 2
        )
        image_y = current_center[..., 1, None] + (
            axis_y[None, None] * self.config.spatial_context * current[..., 3, None] / 2
        )
        grid_x = 2 * (image_x - past_center[..., 0, None]) / (
            self.config.spatial_context * boxes[..., 2, None].clamp_min(1e-5)
        )
        grid_y = 2 * (image_y - past_center[..., 1, None]) / (
            self.config.spatial_context * boxes[..., 3, None].clamp_min(1e-5)
        )
        grid = torch.stack(
            (
                grid_x[:, :, None, :].expand(-1, -1, height, -1),
                grid_y[:, :, :, None].expand(-1, -1, -1, width),
            ),
            dim=-1,
        ).reshape(batch * frames, height, width, 2)
        aligned = F.grid_sample(
            values.reshape(batch * frames, channels, height, width),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return aligned.reshape(batch, frames, channels, height, width)

    @staticmethod
    def _anchor_geometry(anchor: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        anchor_size = anchor[..., 2:].clamp_min(1e-5)
        current_size = current[..., 2:].clamp_min(1e-5)
        anchor_center = anchor[..., :2] + 0.5 * anchor_size
        current_center = current[..., :2] + 0.5 * current_size
        return torch.cat(
            (
                anchor,
                current,
                torch.log(current_size / anchor_size),
                (current_center - anchor_center) / anchor_size,
            ),
            dim=-1,
        )

    @staticmethod
    def _side_strips(current: torch.Tensor) -> torch.Tensor:
        # Pooling occurs only at the final four-side readout; all preceding
        # 3-D blocks retain the complete spatial volume.
        width = current.shape[-1]
        height = current.shape[-2]
        strip_w = max(1, width // 4)
        strip_h = max(1, height // 4)
        strips = (
            current[..., :strip_w],
            current[..., :strip_h, :],
            current[..., -strip_w:],
            current[..., -strip_h:, :],
        )
        return torch.stack([value.mean(dim=(-2, -1)) for value in strips], dim=1)

    def forward(
        self,
        detector_boxes: torch.Tensor,
        decoder_features: torch.Tensor,
        mask_logits: torch.Tensor,
        detector_queries: torch.Tensor,
        history_valid: torch.Tensor,
        anchor_decoder_feature: torch.Tensor | None = None,
        anchor_mask_logit: torch.Tensor | None = None,
        anchor_query: torch.Tensor | None = None,
        anchor_box: torch.Tensor | None = None,
        *,
        gate_threshold: float = 0.5,
        utility_threshold: float | None = None,
    ) -> dict[str, torch.Tensor]:
        boxes = detector_boxes.detach().float()
        features = decoder_features.detach().float()
        logits = mask_logits.detach().float()
        queries = detector_queries.detach().float()
        valid = history_valid.detach().bool()
        if features.ndim != 5:
            raise ValueError("decoder_features must have shape [B,T,C,H,W]")
        batch, frames, channels, height, width = features.shape
        expected = (
            self.config.history_frames,
            self.config.decoder_channels,
            self.config.spatial_size,
            self.config.spatial_size,
        )
        if (frames, channels, height, width) != expected:
            raise ValueError(f"expected decoder feature shape [B,{expected}], got {features.shape}")
        if logits.shape != (batch, frames, 1, height, width):
            raise ValueError("mask_logits shape does not match decoder feature volume")
        if valid.shape != (batch, frames):
            raise ValueError("history_valid must have shape [B,T]")

        anchor_feature = anchor_logit = anchor_token = anchor_bbox = None
        if self.config.use_first_frame_anchor:
            if any(
                value is None
                for value in (
                    anchor_decoder_feature,
                    anchor_mask_logit,
                    anchor_query,
                    anchor_box,
                )
            ):
                raise ValueError("first-frame anchor inputs are required by this model")
            anchor_feature = anchor_decoder_feature.detach().float()
            anchor_logit = anchor_mask_logit.detach().float()
            anchor_token = anchor_query.detach().float()
            anchor_bbox = anchor_box.detach().float()
            if anchor_feature.shape != (batch, channels, height, width):
                raise ValueError("anchor decoder feature shape mismatch")
            if anchor_logit.shape != (batch, 1, height, width):
                raise ValueError("anchor mask logit shape mismatch")
            if anchor_token.shape != (batch, self.config.query_dim):
                raise ValueError("anchor query shape mismatch")
            if anchor_bbox.shape != (batch, 4):
                raise ValueError("anchor bbox shape mismatch")

        volume = torch.cat((features, logits), dim=2)
        volume = self._align_to_current(volume, boxes)
        volume = volume * valid[:, :, None, None, None]
        early_anchor_volume = None
        if (
            self.config.use_first_frame_anchor
            and self.config.anchor_fusion_mode == ANCHOR_FUSION_EARLY_VOLUME
        ):
            # Reuse the exact history ROI warp by appending the current box as
            # the reference slice.  Only the aligned anchor (slice zero) is
            # retained; it is then presented beside every valid history slice.
            anchor_values = torch.cat((anchor_feature, anchor_logit), dim=1)[:, None]
            anchor_pair = torch.cat(
                (anchor_values, torch.zeros_like(anchor_values)), dim=1
            )
            anchor_pair_boxes = torch.cat(
                (anchor_bbox[:, None], boxes[:, -1:]), dim=1
            )
            aligned_anchor = self._align_to_current(
                anchor_pair, anchor_pair_boxes
            )[:, 0]
            repeated_anchor = aligned_anchor[:, None].expand(
                -1, frames, -1, -1, -1
            )
            early_anchor_volume = torch.cat(
                (repeated_anchor, volume - repeated_anchor), dim=2
            )
            early_anchor_volume = (
                early_anchor_volume * valid[:, :, None, None, None]
            ).permute(0, 2, 1, 3, 4)
        axis = torch.linspace(-1.0, 1.0, width, device=volume.device, dtype=volume.dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        coords = torch.stack((xx, yy))[None, None].expand(batch, frames, -1, -1, -1)
        coords = coords * valid[:, :, None, None, None]
        volume = torch.cat((volume, coords), dim=2).permute(0, 2, 1, 3, 4)
        hidden = self.stem(volume)
        if early_anchor_volume is not None:
            hidden = hidden + self.anchor_volume_projection(early_anchor_volume)
        conditioning = self.geometry_encoder(self._geometry(boxes, valid))
        conditioning = conditioning + self.query_encoder(queries)
        if early_anchor_volume is not None:
            anchor_condition = self.anchor_condition_projection(
                torch.cat(
                    (
                        anchor_token,
                        self._anchor_geometry(anchor_bbox, boxes[:, -1]),
                    ),
                    dim=-1,
                )
            )
            conditioning = conditioning + anchor_condition[:, None]
        temporal_valid = valid[:, None, :, None, None]
        conditioning = conditioning.permute(0, 2, 1)[:, :, :, None, None]
        hidden = hidden + (conditioning + self.time_embedding) * temporal_valid
        hidden = self.final_norm(self.blocks(hidden))

        current = hidden[:, :, -1] + self.position
        tokens = current.flatten(2).transpose(1, 2)
        anchor_gate = current.new_zeros(())
        if (
            self.config.use_first_frame_anchor
            and self.config.anchor_fusion_mode == ANCHOR_FUSION_CROSS_ATTENTION
        ):
            anchor_coords = torch.stack((xx, yy))[None].expand(batch, -1, -1, -1)
            anchor_map = self.anchor_stem(
                torch.cat((anchor_feature, anchor_logit, anchor_coords), dim=1)
            )
            anchor_condition = self.anchor_geometry_encoder(
                self._anchor_geometry(anchor_bbox, boxes[:, -1])
            ) + self.query_encoder(anchor_token)
            anchor_map = anchor_map + anchor_condition[:, :, None, None]
            anchor_tokens = anchor_map.flatten(2).transpose(1, 2)
            anchor_match, _ = self.anchor_cross_attention(
                tokens, anchor_tokens, anchor_tokens, need_weights=False
            )
            anchor_gate = torch.tanh(self.anchor_residual_gate)
            tokens = tokens + anchor_gate * self.anchor_cross_norm(anchor_match)
            current = tokens.transpose(1, 2).reshape(batch, self.config.width, height, width)
        elif early_anchor_volume is not None:
            # Diagnostic only: early fusion has no trainable gate and is always
            # part of the single shared 3-D computation.
            anchor_gate = current.new_ones(())
        side_query = self.side_queries.expand(batch, -1, -1)
        attended, _ = self.side_attention(side_query, tokens, tokens, need_weights=False)
        strips = self._side_strips(current)
        global_feature = current.mean(dim=(-2, -1))[:, None].expand(-1, 4, -1)
        side_hidden = self.side_trunk(torch.cat((attended, strips, global_feature), dim=-1))
        side_logits = self.side_head(side_hidden).squeeze(-1)
        side_probability = torch.sigmoid(side_logits)
        raw_expansion = self.expansion_head(side_hidden).squeeze(-1)
        if self.config.expansion_parameterization == BOUNDED_RELATIVE_EXPANSION:
            # Each side predicts a bounded fraction of the current segment-box
            # width/height.  Classification confidence selects the side; it
            # must not shrink a confidently selected geometric magnitude a
            # second time at inference.
            magnitude = torch.sigmoid(raw_expansion) * self.config.max_expansion
            expansion = magnitude
            soft_expansion = side_probability.pow(self.config.gate_power) * magnitude
        elif self.config.expansion_parameterization == LOG_RELATIVE_EXPANSION:
            # Predict uniformly in log(1 + ratio), retaining resolution near
            # zero while making the 2-4x range reachable for tiny fragments.
            magnitude = torch.expm1(
                torch.sigmoid(raw_expansion) * math.log1p(self.config.max_expansion)
            )
            expansion = magnitude
            soft_expansion = side_probability.pow(self.config.gate_power) * magnitude
        else:
            # Preserve v1 checkpoint semantics bit-for-bit.
            magnitude = F.softplus(raw_expansion) - F.softplus(
                torch.tensor(-4.0, device=raw_expansion.device, dtype=raw_expansion.dtype)
            )
            magnitude = magnitude.clamp(0.0, self.config.max_expansion)
            expansion = side_probability.pow(self.config.gate_power) * magnitude
            soft_expansion = expansion
        reference = boxes[:, -1]
        corrected = decode_outward_expansion(
            reference, soft_expansion, maximum=self.config.max_expansion
        )
        edge_active = side_probability >= gate_threshold
        gate_active = edge_active.any(-1)
        hard_edge_expansion = torch.where(edge_active, expansion, torch.zeros_like(expansion))
        hard_decoded = decode_outward_expansion(
            reference, hard_edge_expansion, maximum=self.config.max_expansion
        )
        utility_active = torch.ones_like(gate_active)
        utility_logits = utility_probability = None
        if self.config.predict_correction_utility:
            # Detaching here makes the router a strict consumer of the frozen
            # bbox proposal. Utility supervision can never distort the side or
            # expansion heads, even when a broader trainable scope is used.
            normalized_magnitude = magnitude / self.config.max_expansion
            utility_features = torch.cat(
                (
                    side_hidden.detach().flatten(1),
                    side_probability.detach(),
                    normalized_magnitude.detach(),
                ),
                dim=-1,
            )
            utility_logits = self.correction_utility_head(utility_features).squeeze(-1)
            utility_probability = torch.sigmoid(utility_logits)
            selected_utility_threshold = (
                self.config.correction_utility_threshold
                if utility_threshold is None
                else utility_threshold
            )
            utility_active = utility_probability >= selected_utility_threshold
        side_gate_active = gate_active
        gate_active = side_gate_active & utility_active
        output_boxes = torch.where(gate_active[:, None], hard_decoded, reference)
        result = {
            "side_logits": side_logits,
            "side_probability": side_probability,
            "raw_expansion": raw_expansion,
            "magnitude": magnitude,
            "edge_expansion": expansion,
            "soft_edge_expansion": soft_expansion,
            "hard_edge_expansion": hard_edge_expansion,
            "candidate_boxes": hard_decoded,
            "edge_active": edge_active,
            "reference_boxes": reference,
            "corrected_boxes": corrected,
            "gate_active": gate_active,
            "side_gate_active": side_gate_active,
            "utility_active": utility_active,
            "anchor_residual_gate": anchor_gate,
            "boxes": output_boxes,
        }
        if self.config.predict_measurement_reliability:
            reliability_logits = self.measurement_reliability_head(
                side_hidden.mean(dim=1)
            ).squeeze(-1)
            result["measurement_reliability_logits"] = reliability_logits
            result["measurement_reliability"] = torch.sigmoid(reliability_logits)
        if self.config.predict_correction_utility:
            result["correction_utility_logits"] = utility_logits
            result["correction_utility"] = utility_probability
        return result
