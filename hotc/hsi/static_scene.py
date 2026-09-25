"""Conservative bbox identity lock for persistent one-sided merges."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .motion import AffineMotion, estimate_affine_motion


@dataclass(frozen=True)
class StaticBBoxContainmentPolicy:
    warmup_frames: int = 10
    static_translation_p90: float = 0.5
    static_scale_percent_p90: float = 0.2
    static_rotation_degrees_p90: float = 0.1
    initial_area_ratio: float = 1.50
    initial_anchor_coverage: float = 0.90
    initial_unilateral_score: float = 0.75
    persistent_area_ratio: float = 1.35
    persistent_anchor_coverage: float = 0.85
    confirmation_frames: int = 3
    cooldown_frames: int = 3
    maximum_memory_resets: int = 3


@dataclass(frozen=True)
class StaticBBoxContainmentPolicyV2(StaticBBoxContainmentPolicy):
    maximum_aspect_ratio_change: float = 1.35
    maximum_memory_resets: int = 1


@dataclass(frozen=True)
class StaticBBoxContainmentPolicyV3(StaticBBoxContainmentPolicyV2):
    minimum_anchor_appearance: float = 0.80
    minimum_anchor_appearance_advantage: float = 0.10


@dataclass(frozen=True)
class StaticBBoxContainmentDecision:
    restart: bool
    reason: str
    scene_static: bool | None
    native_box: tuple[float, float, float, float]
    anchor_box: tuple[float, float, float, float] | None
    previous_area_ratio: float
    previous_coverage: float
    anchor_area_ratio: float
    anchor_coverage: float
    unilateral_score: float
    aspect_ratio_change: float
    anchor_appearance_score: float | None
    native_appearance_score: float | None
    confirmation_count: int
    restart_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def intersection_over_reference(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    near = np.maximum(reference[:2], candidate[:2])
    far = np.minimum(reference[:2] + reference[2:], candidate[:2] + candidate[2:])
    intersection = float(np.maximum(far - near, 0).prod())
    return intersection / max(float(reference[2] * reference[3]), 1.0)


def unilateral_expansion_score(anchor: np.ndarray, candidate: np.ndarray) -> float:
    anchor = np.asarray(anchor, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    anchor_far = anchor[:2] + anchor[2:]
    candidate_far = candidate[:2] + candidate[2:]
    left_top = np.maximum(anchor[:2] - candidate[:2], 0.0)
    right_bottom = np.maximum(candidate_far - anchor_far, 0.0)
    expansion = np.concatenate((left_top, right_bottom))
    denominator = float(expansion.sum())
    if denominator <= 0:
        return 0.0
    x_imbalance = abs(float(left_top[0] - right_bottom[0]))
    y_imbalance = abs(float(left_top[1] - right_bottom[1]))
    return (x_imbalance + y_imbalance) / denominator


def aspect_ratio_change(anchor: np.ndarray, candidate: np.ndarray) -> float:
    anchor = np.asarray(anchor, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    anchor_aspect = max(float(anchor[2] / max(anchor[3], 1.0)), 1e-6)
    candidate_aspect = max(float(candidate[2] / max(candidate[3], 1.0)), 1e-6)
    ratio = candidate_aspect / anchor_aspect
    return max(ratio, 1.0 / ratio)


def box_mask(
    shape: tuple[int, int], box: np.ndarray | tuple[float, ...]
) -> np.ndarray:
    height, width = shape
    x, y, box_width, box_height = np.asarray(box, dtype=np.float64)
    left = max(0, min(int(np.floor(x)), width - 1))
    top = max(0, min(int(np.floor(y)), height - 1))
    right = max(left + 1, min(int(np.ceil(x + box_width)), width))
    bottom = max(top + 1, min(int(np.ceil(y + box_height)), height))
    result = np.zeros((height, width), dtype=bool)
    result[top:bottom, left:right] = True
    return result


def clip_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    x, y, box_width, box_height = np.asarray(box, dtype=np.float64)
    x = float(np.clip(x, 0, max(width - 1, 0)))
    y = float(np.clip(y, 0, max(height - 1, 0)))
    box_width = float(np.clip(box_width, 1, max(width - x, 1)))
    box_height = float(np.clip(box_height, 1, max(height - y, 1)))
    return np.asarray((x, y, box_width, box_height), dtype=np.float64)


def appearance_patch(image: np.ndarray, box: np.ndarray, size: int = 32) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim == 2:
        value = value[..., None]
    height, width = value.shape[:2]
    x, y, box_width, box_height = np.asarray(box, dtype=np.float64)
    left = max(0, min(int(np.floor(x)), width - 1))
    top = max(0, min(int(np.floor(y)), height - 1))
    right = max(left + 1, min(int(np.ceil(x + box_width)), width))
    bottom = max(top + 1, min(int(np.ceil(y + box_height)), height))
    patch = value[top:bottom, left:right]
    return np.asarray(
        cv2.resize(patch, (size, size), interpolation=cv2.INTER_CUBIC),
        dtype=np.float32,
    ) / 255.0


def appearance_zncc(reference: np.ndarray, candidate: np.ndarray) -> float:
    left = np.asarray(reference, dtype=np.float32)
    right = np.asarray(candidate, dtype=np.float32)
    if left.shape != right.shape:
        raise ValueError("appearance patches must have the same shape")
    axes = tuple(range(left.ndim - 1)) if left.ndim > 2 else tuple(range(left.ndim))
    left = left - left.mean(axis=axes, keepdims=True)
    right = right - right.mean(axis=axes, keepdims=True)
    denominator = float(np.sqrt(np.square(left).sum() * np.square(right).sum()))
    if denominator <= 1e-8:
        return 0.0
    return float((left * right).sum() / denominator)


class StaticBBoxContainmentGate:
    """Reprompt only after a persistent nested, one-sided bbox expansion."""

    def __init__(
        self,
        policy: StaticBBoxContainmentPolicy = StaticBBoxContainmentPolicy(),
    ) -> None:
        self.policy = policy
        self.previous_box: np.ndarray | None = None
        self.anchor_box: np.ndarray | None = None
        self.global_motion: list[AffineMotion] = []
        self.confirmation_count = 0
        self.last_restart = -10**9
        self.restart_count = 0
        self.anchor_reference_patch: np.ndarray | None = None

    def initialize(self, box: tuple[float, float, float, float]) -> None:
        self.previous_box = np.asarray(box, dtype=np.float64)
        self.anchor_box = None
        self.global_motion = []
        self.confirmation_count = 0
        self.last_restart = -10**9
        self.restart_count = 0
        self.anchor_reference_patch = None

    def observe_global_pair(
        self,
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
        previous_region: np.ndarray | None,
    ) -> AffineMotion | None:
        if len(self.global_motion) >= self.policy.warmup_frames:
            return None
        motion = estimate_affine_motion(
            previous_gray,
            current_gray,
            mask=previous_region,
            exclude_mask=True,
        )
        if motion is not None:
            self.global_motion.append(motion)
        return motion

    @property
    def scene_static(self) -> bool | None:
        if len(self.global_motion) < self.policy.warmup_frames:
            return None
        translation = np.percentile([row.translation for row in self.global_motion], 90)
        scale = np.percentile([abs(row.scale - 1.0) * 100 for row in self.global_motion], 90)
        rotation = np.percentile([abs(row.rotation_degrees) for row in self.global_motion], 90)
        return bool(
            translation < self.policy.static_translation_p90
            and scale < self.policy.static_scale_percent_p90
            and rotation < self.policy.static_rotation_degrees_p90
        )

    def scene_diagnostics(self) -> dict[str, object]:
        if not self.global_motion:
            return {"pairs": 0, "static": self.scene_static}
        return {
            "pairs": len(self.global_motion),
            "static": self.scene_static,
            "translation_p90": float(np.percentile([row.translation for row in self.global_motion], 90)),
            "scale_percent_p90": float(
                np.percentile([abs(row.scale - 1.0) * 100 for row in self.global_motion], 90)
            ),
            "rotation_degrees_p90": float(
                np.percentile([abs(row.rotation_degrees) for row in self.global_motion], 90)
            ),
        }

    def inspect(
        self,
        native_box: tuple[float, float, float, float],
        frame_index: int,
        previous_image: np.ndarray | None = None,
        current_image: np.ndarray | None = None,
    ) -> StaticBBoxContainmentDecision:
        if self.previous_box is None:
            raise RuntimeError("gate must be initialized")
        native = np.asarray(native_box, dtype=np.float64)
        previous = self.previous_box
        previous_ratio = float(native[2] * native[3]) / max(float(previous[2] * previous[3]), 1.0)
        previous_coverage = intersection_over_reference(previous, native)
        initial_unilateral = unilateral_expansion_score(previous, native)
        aspect_change = aspect_ratio_change(previous, native)
        maximum_aspect_change = float(
            getattr(self.policy, "maximum_aspect_ratio_change", float("inf"))
        )
        minimum_anchor_appearance = float(
            getattr(self.policy, "minimum_anchor_appearance", -float("inf"))
        )
        minimum_appearance_advantage = float(
            getattr(self.policy, "minimum_anchor_appearance_advantage", -float("inf"))
        )
        uses_appearance = isinstance(self.policy, StaticBBoxContainmentPolicyV3)
        reference_patch = None
        anchor_appearance_score = None
        native_appearance_score = None
        if uses_appearance:
            if previous_image is None or current_image is None:
                raise ValueError("appearance policy requires previous and current images")
            reference_patch = appearance_patch(previous_image, previous)
            anchor_appearance_score = appearance_zncc(
                reference_patch, appearance_patch(current_image, previous)
            )
            native_appearance_score = appearance_zncc(
                reference_patch, appearance_patch(current_image, native)
            )
        appearance_allowed = bool(
            not uses_appearance
            or (
                anchor_appearance_score is not None
                and native_appearance_score is not None
                and anchor_appearance_score >= minimum_anchor_appearance
                and anchor_appearance_score - native_appearance_score
                >= minimum_appearance_advantage
            )
        )
        scene_static = self.scene_static
        in_cooldown = frame_index - self.last_restart <= self.policy.cooldown_frames
        may_start = bool(
            scene_static is True
            and not in_cooldown
            and self.restart_count < self.policy.maximum_memory_resets
            and previous_ratio >= self.policy.initial_area_ratio
            and previous_coverage >= self.policy.initial_anchor_coverage
            and initial_unilateral >= self.policy.initial_unilateral_score
            and aspect_change <= maximum_aspect_change
            and appearance_allowed
        )
        anchor_ratio = 1.0
        anchor_coverage = 1.0
        unilateral = initial_unilateral
        if self.anchor_box is None:
            if may_start:
                self.anchor_box = previous.copy()
                self.anchor_reference_patch = reference_patch
                self.confirmation_count = 1
                reason = "nested_expansion_confirming"
            else:
                self.confirmation_count = 0
                reason = "bbox_passthrough"
        else:
            anchor_area = max(float(self.anchor_box[2] * self.anchor_box[3]), 1.0)
            anchor_ratio = float(native[2] * native[3]) / anchor_area
            anchor_coverage = intersection_over_reference(self.anchor_box, native)
            unilateral = unilateral_expansion_score(self.anchor_box, native)
            aspect_change = aspect_ratio_change(self.anchor_box, native)
            if uses_appearance:
                if current_image is None or self.anchor_reference_patch is None:
                    raise ValueError("appearance confirmation lost its reference patch")
                anchor_appearance_score = appearance_zncc(
                    self.anchor_reference_patch,
                    appearance_patch(current_image, self.anchor_box),
                )
                native_appearance_score = appearance_zncc(
                    self.anchor_reference_patch,
                    appearance_patch(current_image, native),
                )
                appearance_allowed = bool(
                    anchor_appearance_score >= minimum_anchor_appearance
                    and anchor_appearance_score - native_appearance_score
                    >= minimum_appearance_advantage
                )
            persists = bool(
                anchor_ratio >= self.policy.persistent_area_ratio
                and anchor_coverage >= self.policy.persistent_anchor_coverage
                and unilateral >= self.policy.initial_unilateral_score
                and aspect_change <= maximum_aspect_change
                and appearance_allowed
            )
            if persists:
                self.confirmation_count += 1
                reason = "nested_expansion_confirming"
            else:
                self.anchor_box = None
                self.anchor_reference_patch = None
                self.confirmation_count = 0
                reason = "nested_expansion_cancelled"
        restart = bool(
            self.anchor_box is not None
            and self.confirmation_count >= self.policy.confirmation_frames
        )
        decision_anchor = None if self.anchor_box is None else tuple(map(float, self.anchor_box))
        if restart:
            reason = "bbox_reprompt_persistent_nested_expansion"
            self.last_restart = frame_index
            self.restart_count += 1
            self.previous_box = self.anchor_box.copy()
            self.anchor_box = None
            self.anchor_reference_patch = None
            self.confirmation_count = 0
        else:
            self.previous_box = native.copy()
        return StaticBBoxContainmentDecision(
            restart=restart,
            reason=reason,
            scene_static=scene_static,
            native_box=tuple(map(float, native)),
            anchor_box=decision_anchor,
            previous_area_ratio=previous_ratio,
            previous_coverage=previous_coverage,
            anchor_area_ratio=anchor_ratio,
            anchor_coverage=anchor_coverage,
            unilateral_score=unilateral,
            aspect_ratio_change=aspect_change,
            anchor_appearance_score=anchor_appearance_score,
            native_appearance_score=native_appearance_score,
            confirmation_count=self.confirmation_count,
            restart_count=self.restart_count,
        )
