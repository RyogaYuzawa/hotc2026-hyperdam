"""Frame-zero spectral reference model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .hyperspectral import _unit_shape, ring_mask


@dataclass(frozen=True)
class ReferenceMatch:
    accepted: bool
    similarity: float
    matched_fraction: float
    reliable: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _hellinger_similarity(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    distance = np.sqrt(0.5 * np.sum((np.sqrt(first) - np.sqrt(second)) ** 2))
    return float(np.clip(1.0 - distance, 0.0, 1.0))


def _fit_prototypes(
    spectra: np.ndarray,
    clusters: int,
    *,
    random_state: int,
    max_samples: int,
) -> np.ndarray:
    from sklearn.cluster import MiniBatchKMeans

    spectra = _unit_shape(spectra)
    generator = np.random.default_rng(random_state)
    if len(spectra) > max_samples:
        spectra = spectra[generator.choice(len(spectra), max_samples, replace=False)]
    cluster_count = max(1, min(int(clusters), len(spectra)))
    if cluster_count == 1:
        center = np.mean(spectra, axis=0, keepdims=True)
    else:
        center = MiniBatchKMeans(
            n_clusters=cluster_count,
            batch_size=min(1024, len(spectra)),
            n_init=5,
            random_state=random_state,
        ).fit(spectra).cluster_centers_
    return _unit_shape(center).astype(np.float32)


class MultiPrototypeSpectralReference:
    """Frame-zero spectral fingerprint for heterogeneous bbox contents.

    A final histogram bin represents pixels that are not closer to any target
    prototype than to the local background prototypes. Thus a candidate made
    from only one matching material cannot look identical to a multi-material
    reference merely because every pixel is assigned to its nearest cluster.
    """

    def __init__(
        self,
        target_prototypes: np.ndarray,
        background_prototypes: np.ndarray,
        margin_threshold: float,
        reference_histogram: np.ndarray,
        ring_similarity: float,
        reference_matched_fraction: float,
        ring_matched_fraction: float,
        target_pixels: int,
        *,
        similarity_retention: float = 0.50,
        containment_retention: float = 0.10,
        min_reference_separation: float = 0.10,
    ):
        self.target_prototypes = np.asarray(target_prototypes, dtype=np.float32)
        self.background_prototypes = np.asarray(
            background_prototypes, dtype=np.float32
        )
        self.margin_threshold = float(margin_threshold)
        self.reference_histogram = np.asarray(reference_histogram, dtype=np.float64)
        self.ring_similarity = float(ring_similarity)
        self.reference_matched_fraction = float(reference_matched_fraction)
        self.ring_matched_fraction = float(ring_matched_fraction)
        self.initial_matched_advantage = float(
            reference_matched_fraction - ring_matched_fraction
        )
        self.target_pixels = int(target_pixels)
        self.similarity_retention = float(similarity_retention)
        self.containment_retention = float(containment_retention)
        self.min_reference_separation = float(min_reference_separation)
        self.similarity_floor = float(
            ring_similarity + similarity_retention * (1.0 - ring_similarity)
        )
        self.matched_fraction_floor = float(
            ring_matched_fraction
            + containment_retention * self.initial_matched_advantage
        )
        self.reliable = bool(
            target_pixels >= max(16, self.target_prototypes.shape[1])
            and self.initial_matched_advantage >= min_reference_separation
        )

    @classmethod
    def fit(
        cls,
        cube: np.ndarray,
        target_mask: np.ndarray,
        *,
        target_clusters: int = 4,
        background_clusters: int = 4,
        random_state: int = 2026,
        max_samples: int = 8192,
        similarity_retention: float = 0.50,
        containment_retention: float = 0.10,
        min_reference_separation: float = 0.10,
    ) -> "MultiPrototypeSpectralReference":
        ring = ring_mask(target_mask, scale=1.5)
        target = cube[target_mask]
        background = cube[ring]
        target_prototypes = _fit_prototypes(
            target,
            target_clusters,
            random_state=random_state,
            max_samples=max_samples,
        )
        background_prototypes = _fit_prototypes(
            background,
            background_clusters,
            random_state=random_state + 1,
            max_samples=max_samples,
        )

        provisional = cls(
            target_prototypes,
            background_prototypes,
            margin_threshold=0.0,
            reference_histogram=np.ones(len(target_prototypes) + 1)
            / (len(target_prototypes) + 1),
            ring_similarity=0.0,
            reference_matched_fraction=0.75,
            ring_matched_fraction=0.0,
            target_pixels=len(target),
            similarity_retention=similarity_retention,
            containment_retention=containment_retention,
            min_reference_separation=min_reference_separation,
        )
        target_margins = provisional._margins(target)
        margin_threshold = float(np.quantile(target_margins, 0.25))
        provisional.margin_threshold = margin_threshold
        reference_histogram, reference_matched = provisional._histogram(target)
        provisional.reference_histogram = reference_histogram
        ring_histogram, ring_matched = provisional._histogram(background)
        ring_similarity = _hellinger_similarity(reference_histogram, ring_histogram)
        return cls(
            target_prototypes,
            background_prototypes,
            margin_threshold,
            reference_histogram,
            ring_similarity,
            reference_matched,
            ring_matched,
            len(target),
            similarity_retention=similarity_retention,
            containment_retention=containment_retention,
            min_reference_separation=min_reference_separation,
        )

    def _scores(self, spectra: np.ndarray):
        shaped = _unit_shape(spectra)
        target = shaped @ self.target_prototypes.T
        background = shaped @ self.background_prototypes.T
        assignments = np.argmax(target, axis=1)
        margins = np.max(target, axis=1) - np.max(background, axis=1)
        return assignments, margins

    def _margins(self, spectra: np.ndarray) -> np.ndarray:
        return self._scores(spectra)[1]

    def _histogram(self, spectra: np.ndarray) -> tuple[np.ndarray, float]:
        assignments, margins = self._scores(spectra)
        matched = margins >= self.margin_threshold
        counts = np.bincount(
            assignments[matched], minlength=len(self.target_prototypes)
        ).astype(np.float64)
        histogram = np.concatenate((counts, [np.count_nonzero(~matched)]))
        histogram /= max(float(np.sum(histogram)), 1.0)
        return histogram, float(np.mean(matched))

    def evaluate(self, cube: np.ndarray, mask: np.ndarray) -> ReferenceMatch:
        if cube.shape[:2] != mask.shape or not np.any(mask):
            return ReferenceMatch(False, 0.0, 0.0, self.reliable)
        histogram, matched_fraction = self._histogram(cube[mask])
        similarity = _hellinger_similarity(self.reference_histogram, histogram)
        # This is intentionally one-sided containment, not exact mixture
        # equality: occlusion may hide some initial materials, but a candidate
        # should still be explainable by the frame-zero target prototype set.
        accepted = bool(
            not self.reliable or matched_fraction >= self.matched_fraction_floor
        )
        return ReferenceMatch(
            accepted=accepted,
            similarity=similarity,
            matched_fraction=matched_fraction,
            reliable=self.reliable,
        )

    def calibration_dict(self) -> dict:
        return {
            "target_prototypes": len(self.target_prototypes),
            "background_prototypes": len(self.background_prototypes),
            "margin_threshold": self.margin_threshold,
            "reference_histogram": self.reference_histogram.tolist(),
            "ring_similarity": self.ring_similarity,
            "similarity_floor": self.similarity_floor,
            "similarity_retention": self.similarity_retention,
            "reference_matched_fraction": self.reference_matched_fraction,
            "ring_matched_fraction": self.ring_matched_fraction,
            "initial_matched_advantage": self.initial_matched_advantage,
            "matched_fraction_floor": self.matched_fraction_floor,
            "containment_retention": self.containment_retention,
            "min_reference_separation": self.min_reference_separation,
            "target_pixels": self.target_pixels,
            "reliable": self.reliable,
        }
