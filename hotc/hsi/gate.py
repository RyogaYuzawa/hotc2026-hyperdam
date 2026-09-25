"""Hyperspectral identity gate for SAM3 memory updates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from .hyperspectral import SpectralModel, ring_mask, spectral_angle_similarity
from .reference import MultiPrototypeSpectralReference


@dataclass(frozen=True)
class HSIMatch:
    accepted: bool
    prototype_similarity: float
    evidence_fraction: float
    upper_evidence: float
    local_contrast: float
    normalized_score: float
    spatial_auc: float = 0.5
    evidence_advantage: float = 0.0
    reliable: bool = True
    abstained: bool = False
    rejection_reasons: tuple[str, ...] = ()
    reference_similarity: float = 1.0
    reference_matched_fraction: float = 1.0
    reference_reliable: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class HSIIdentityGate:
    """A frame-zero calibrated gate for DAM/DRM memory writes."""

    def __init__(
        self,
        model: SpectralModel,
        evidence_threshold: float,
        initial_upper_evidence: float,
        initial_local_contrast: float,
        *,
        min_prototype_similarity: float = 0.96,
        min_evidence_fraction: float = 0.20,
        min_upper_ratio: float = 0.85,
        min_local_contrast: float | None = None,
        min_local_contrast_ratio: float = 0.25,
    ):
        self.model = model
        self.evidence_threshold = float(evidence_threshold)
        self.initial_upper_evidence = float(initial_upper_evidence)
        self.initial_local_contrast = float(initial_local_contrast)
        self.min_prototype_similarity = float(min_prototype_similarity)
        self.min_evidence_fraction = float(min_evidence_fraction)
        self.min_upper_ratio = float(min_upper_ratio)
        self.min_local_contrast_ratio = float(min_local_contrast_ratio)
        self.min_local_contrast = float(
            min_local_contrast
            if min_local_contrast is not None
            else max(0.0, initial_local_contrast * min_local_contrast_ratio)
        )

    @staticmethod
    def _upper_mean(values: np.ndarray, fraction: float = 0.25) -> float:
        count = max(1, int(np.ceil(len(values) * fraction)))
        return float(np.mean(np.partition(values, len(values) - count)[-count:]))

    @classmethod
    def fit(cls, cube: np.ndarray, target_mask: np.ndarray, **kwargs) -> "HSIIdentityGate":
        model = SpectralModel.fit(cube, target_mask)
        heatmap = model.heatmap(cube)
        evidence = heatmap[target_mask]
        initial_upper = cls._upper_mean(evidence)
        try:
            initial_ring = ring_mask(target_mask, scale=1.5)
            ring_upper = cls._upper_mean(heatmap[initial_ring])
        except Exception:
            ring_upper = initial_upper
        return cls(
            model,
            evidence_threshold=float(np.quantile(evidence, 0.25)),
            initial_upper_evidence=initial_upper,
            initial_local_contrast=initial_upper - ring_upper,
            **kwargs,
        )

    def evaluate(self, cube: np.ndarray, mask: np.ndarray) -> HSIMatch:
        if cube.shape[:2] != mask.shape or not np.any(mask):
            return HSIMatch(False, 0.0, 0.0, 0.0, -1.0, 0.0)
        observation = np.median(cube[mask], axis=0)
        prototype_similarity = float(
            spectral_angle_similarity(
                observation.reshape(1, 1, -1), self.model.target_prototype
            )[0, 0]
        )
        heatmap = self.model.heatmap(cube)
        evidence = heatmap[mask]
        evidence_fraction = float(np.mean(evidence >= self.evidence_threshold))
        upper_evidence = self._upper_mean(evidence)
        try:
            ring = ring_mask(mask, scale=1.5)
            ring_upper = self._upper_mean(heatmap[ring]) if np.any(ring) else upper_evidence
        except Exception:
            ring_upper = upper_evidence
        local_contrast = upper_evidence - ring_upper
        upper_floor = self.min_upper_ratio * self.initial_upper_evidence
        ratios = (
            prototype_similarity / max(self.min_prototype_similarity, 1e-6),
            evidence_fraction / max(self.min_evidence_fraction, 1e-6),
            upper_evidence / max(upper_floor, 1e-6),
        )
        normalized_score = float(min(ratios))
        accepted = bool(
            prototype_similarity >= self.min_prototype_similarity
            and evidence_fraction >= self.min_evidence_fraction
            and upper_evidence >= upper_floor
            and local_contrast >= self.min_local_contrast
        )
        return HSIMatch(
            accepted,
            prototype_similarity,
            evidence_fraction,
            upper_evidence,
            local_contrast,
            normalized_score,
        )

    def calibration_dict(self) -> dict:
        return {
            "evidence_threshold": self.evidence_threshold,
            "initial_upper_evidence": self.initial_upper_evidence,
            "initial_local_contrast": self.initial_local_contrast,
            "min_prototype_similarity": self.min_prototype_similarity,
            "min_evidence_fraction": self.min_evidence_fraction,
            "min_upper_ratio": self.min_upper_ratio,
            "min_local_contrast": self.min_local_contrast,
            "min_local_contrast_ratio": self.min_local_contrast_ratio,
        }


def _rank_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Tie-aware probability that a positive score exceeds a negative score."""
    positive = np.asarray(positive, dtype=np.float64).reshape(-1)
    negative = np.asarray(negative, dtype=np.float64).reshape(-1)
    if not len(positive) or not len(negative):
        return 0.5
    values = np.concatenate((positive, negative))
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    rank_sum = float(np.sum(ranks[: len(positive)]))
    return float(
        (rank_sum - len(positive) * (len(positive) + 1) / 2)
        / (len(positive) * len(negative))
    )


class CalibratedHSIIdentityGate(HSIIdentityGate):
    """A distributional DRM gate that can abstain when frame-zero HSI is weak.

    V1 uses absolute statistics of the proposed mask. V2 also asks whether the
    proposed mask is spectrally better than its *current-frame* local ring. The
    latter is invariant to much of the radiometric drift between frames. Feature
    floors are derived only from frame zero; no dataset or GT labels are used.
    """

    def __init__(
        self,
        *args,
        initial_spatial_auc: float,
        initial_evidence_fraction: float,
        initial_ring_evidence_fraction: float,
        target_pixels: int,
        bands: int,
        min_initial_spatial_auc: float = 0.60,
        min_initial_evidence_advantage: float = 0.10,
        spatial_retention: float = 0.35,
        advantage_retention: float = 0.10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.initial_spatial_auc = float(initial_spatial_auc)
        self.initial_evidence_fraction = float(initial_evidence_fraction)
        self.initial_ring_evidence_fraction = float(initial_ring_evidence_fraction)
        self.initial_evidence_advantage = float(
            initial_evidence_fraction - initial_ring_evidence_fraction
        )
        self.target_pixels = int(target_pixels)
        self.bands = int(bands)
        self.min_initial_spatial_auc = float(min_initial_spatial_auc)
        self.min_initial_evidence_advantage = float(min_initial_evidence_advantage)
        self.spatial_retention = float(spatial_retention)
        self.advantage_retention = float(advantage_retention)
        self.reliable = bool(
            self.target_pixels >= max(12, self.bands)
            and self.initial_spatial_auc >= self.min_initial_spatial_auc
            and self.initial_evidence_advantage
            >= self.min_initial_evidence_advantage
        )

    @classmethod
    def fit(cls, cube: np.ndarray, target_mask: np.ndarray, **kwargs):
        model = SpectralModel.fit(cube, target_mask)
        heatmap = model.heatmap(cube)
        target_evidence = heatmap[target_mask]
        evidence_threshold = float(np.quantile(target_evidence, 0.25))
        initial_ring = ring_mask(target_mask, scale=1.5)
        ring_evidence = heatmap[initial_ring]
        initial_upper = cls._upper_mean(target_evidence)
        ring_upper = cls._upper_mean(ring_evidence)
        initial_fraction = float(np.mean(target_evidence >= evidence_threshold))
        initial_ring_fraction = float(np.mean(ring_evidence >= evidence_threshold))
        return cls(
            model,
            evidence_threshold=evidence_threshold,
            initial_upper_evidence=initial_upper,
            initial_local_contrast=initial_upper - ring_upper,
            initial_spatial_auc=_rank_auc(target_evidence, ring_evidence),
            initial_evidence_fraction=initial_fraction,
            initial_ring_evidence_fraction=initial_ring_fraction,
            target_pixels=int(np.count_nonzero(target_mask)),
            bands=int(cube.shape[-1]),
            **kwargs,
        )

    def evaluate(self, cube: np.ndarray, mask: np.ndarray) -> HSIMatch:
        base = super().evaluate(cube, mask)
        if cube.shape[:2] != mask.shape or not np.any(mask):
            return HSIMatch(
                False,
                base.prototype_similarity,
                base.evidence_fraction,
                base.upper_evidence,
                base.local_contrast,
                0.0,
                reliable=self.reliable,
                rejection_reasons=("empty_mask",),
            )

        heatmap = self.model.heatmap(cube)
        evidence = heatmap[mask]
        try:
            ring = ring_mask(mask, scale=1.5)
            ring_evidence = heatmap[ring]
        except Exception:
            ring_evidence = np.empty(0, dtype=np.float32)
        spatial_auc = _rank_auc(evidence, ring_evidence)
        ring_fraction = float(
            np.mean(ring_evidence >= self.evidence_threshold)
            if len(ring_evidence)
            else base.evidence_fraction
        )
        evidence_advantage = float(base.evidence_fraction - ring_fraction)

        # Retain only a fraction of frame-zero separation. This is stricter than
        # an absolute score but allows global illumination to change over time.
        spatial_floor = 0.5 + self.spatial_retention * (
            self.initial_spatial_auc - 0.5
        )
        advantage_floor = (
            self.advantage_retention * self.initial_evidence_advantage
        )
        coverage_floor = self.initial_ring_evidence_fraction + 0.25 * (
            self.initial_evidence_advantage
        )

        prototype_pass = base.prototype_similarity >= self.min_prototype_similarity
        absolute_pass = bool(
            base.evidence_fraction >= coverage_floor
            and base.upper_evidence
            >= self.min_upper_ratio * self.initial_upper_evidence
        )
        relative_pass = bool(
            spatial_auc >= spatial_floor
            and evidence_advantage >= advantage_floor
        )
        reasons: list[str] = []
        if not prototype_pass:
            reasons.append("prototype")
        # Absolute coverage is fragile under exposure change. A strong current
        # mask-vs-ring margin is an alternative route, rather than another AND.
        if not (absolute_pass or relative_pass):
            reasons.append("spectral_evidence")
        if (
            self.initial_local_contrast > 0.05
            and base.local_contrast < self.min_local_contrast
        ):
            reasons.append("local_contrast")

        abstained = not self.reliable
        accepted = bool(abstained or not reasons)
        absolute_score = min(
            base.evidence_fraction / max(coverage_floor, 1e-6),
            base.upper_evidence
            / max(self.min_upper_ratio * self.initial_upper_evidence, 1e-6),
        )
        relative_score = min(
            spatial_auc / max(spatial_floor, 1e-6),
            evidence_advantage / max(advantage_floor, 1e-6),
        )
        ratios = (
            base.prototype_similarity / max(self.min_prototype_similarity, 1e-6),
            max(absolute_score, relative_score),
        )
        return HSIMatch(
            accepted=accepted,
            prototype_similarity=base.prototype_similarity,
            evidence_fraction=base.evidence_fraction,
            upper_evidence=base.upper_evidence,
            local_contrast=base.local_contrast,
            normalized_score=float(min(ratios)),
            spatial_auc=spatial_auc,
            evidence_advantage=evidence_advantage,
            reliable=self.reliable,
            abstained=abstained,
            rejection_reasons=tuple(reasons),
        )

    def calibration_dict(self) -> dict:
        result = super().calibration_dict()
        result.update(
            {
                "version": 2,
                "initial_spatial_auc": self.initial_spatial_auc,
                "initial_evidence_fraction": self.initial_evidence_fraction,
                "initial_ring_evidence_fraction": self.initial_ring_evidence_fraction,
                "initial_evidence_advantage": self.initial_evidence_advantage,
                "target_pixels": self.target_pixels,
                "bands": self.bands,
                "min_initial_spatial_auc": self.min_initial_spatial_auc,
                "min_initial_evidence_advantage": self.min_initial_evidence_advantage,
                "spatial_retention": self.spatial_retention,
                "advantage_retention": self.advantage_retention,
                "reliable": self.reliable,
            }
        )
        return result


class ReferenceHSIIdentityGate:
    """V3: V2 DRM verification plus a multi-prototype bbox fingerprint."""

    def __init__(
        self,
        base_gate: CalibratedHSIIdentityGate,
        reference: MultiPrototypeSpectralReference,
    ):
        self.base_gate = base_gate
        self.reference = reference

    @classmethod
    def fit(
        cls,
        cube: np.ndarray,
        target_mask: np.ndarray,
        *,
        reference_similarity_retention: float = 0.50,
        reference_containment_retention: float = 0.10,
        min_reference_separation: float = 0.10,
        target_clusters: int = 4,
        background_clusters: int = 4,
        **kwargs,
    ) -> "ReferenceHSIIdentityGate":
        base_gate = CalibratedHSIIdentityGate.fit(cube, target_mask, **kwargs)
        reference = MultiPrototypeSpectralReference.fit(
            cube,
            target_mask,
            target_clusters=target_clusters,
            background_clusters=background_clusters,
            similarity_retention=reference_similarity_retention,
            containment_retention=reference_containment_retention,
            min_reference_separation=min_reference_separation,
        )
        return cls(base_gate, reference)

    def evaluate(self, cube: np.ndarray, mask: np.ndarray) -> HSIMatch:
        base = self.base_gate.evaluate(cube, mask)
        reference = self.reference.evaluate(cube, mask)
        reasons = list(base.rejection_reasons)
        if reference.reliable and not reference.accepted:
            reasons.append("reference_containment")
        accepted = bool(base.accepted and reference.accepted)
        reference_ratio = reference.matched_fraction / max(
            self.reference.matched_fraction_floor, 1e-6
        )
        return replace(
            base,
            accepted=accepted,
            normalized_score=float(min(base.normalized_score, reference_ratio)),
            rejection_reasons=tuple(reasons),
            reference_similarity=reference.similarity,
            reference_matched_fraction=reference.matched_fraction,
            reference_reliable=reference.reliable,
        )

    def calibration_dict(self) -> dict:
        result = self.base_gate.calibration_dict()
        result["version"] = 3
        result["reference"] = self.reference.calibration_dict()
        return result
