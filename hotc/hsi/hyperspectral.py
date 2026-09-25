"""Hyperspectral frame loading and spectral features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


SENSOR_LAYOUTS = {
    "vis": (4, 16),
    "nir": (5, 25),
    "rednir": (4, 15),
}
FALSE_COLOR_TO_RAW = {
    "HSI-VIS-FalseColor_25": "HSI-VIS",
    "HSI-VIS-FalseColor": "HSI-VIS",
    "HSI-NIR-FalseColor": "HSI-NIR",
    "HSI-RedNIR-FalseColor": "HSI-RedNIR",
}


class HyperspectralError(RuntimeError):
    pass


def raw_hsi_path(false_color_path: Path) -> Path:
    """Resolve the strictly aligned raw snapshot mosaic for a false-color frame."""
    value = str(Path(false_color_path))
    for false_color, raw in FALSE_COLOR_TO_RAW.items():
        token = f"/{false_color}/"
        if token in value:
            result = Path(value.replace(token, f"/{raw}/")).with_suffix(".png")
            candidates = [result]
            if result.is_file():
                return result
            # Some official update archives add different redundant directories
            # to false-color and raw streams (for example ``park/img/park`` versus
            # ``park/img``). The frame stem remains aligned. Search only inside
            # this sequence's raw root and accept the fallback only when unique.
            matches: list[Path] = []
            raw_token = f"/{raw}/"
            prefix, relative = str(result).split(raw_token, 1)
            relative_parts = Path(relative).parts
            if relative_parts:
                sequence_root = Path(prefix) / raw / relative_parts[0]
                if sequence_root.is_dir():
                    matches.extend(sequence_root.rglob(result.name))
            unique_matches = sorted(set(matches))
            if len(unique_matches) == 1:
                return unique_matches[0]
            candidates.extend(unique_matches)
            raise HyperspectralError(
                "raw HSI frame is missing; checked: "
                + ", ".join(map(str, candidates))
            )
    raise HyperspectralError(f"cannot infer raw HSI modality from {false_color_path}")


def mosaic_to_cube(image: np.ndarray, sensor: str) -> np.ndarray:
    """Convert an X2Cube mosaic to its aligned H x W x bands cube."""
    if sensor not in SENSOR_LAYOUTS:
        raise HyperspectralError(f"unsupported sensor: {sensor}")
    if image.ndim != 2:
        raise HyperspectralError(f"expected a grayscale mosaic, got {image.shape}")
    block, useful_bands = SENSOR_LAYOUTS[sensor]
    height, width = image.shape
    if height % block or width % block:
        raise HyperspectralError(
            f"mosaic shape {image.shape} is not divisible by sensor block {block}"
        )
    cube = (
        image.reshape(height // block, block, width // block, block)
        .transpose(0, 2, 1, 3)
        .reshape(height // block, width // block, block * block)
    )
    return cube[..., :useful_bands]


def load_cube(false_color_path: Path, sensor: str) -> np.ndarray:
    path = raw_hsi_path(false_color_path)
    mosaic = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mosaic is None:
        raise HyperspectralError(f"could not read raw HSI frame: {path}")
    return mosaic_to_cube(mosaic, sensor).astype(np.float32)


def _unit_shape(spectra: np.ndarray) -> np.ndarray:
    """Remove amplitude/offset and normalize spectral shape for angle matching."""
    spectra = np.asarray(spectra, dtype=np.float32)
    centered = spectra - spectra.mean(axis=-1, keepdims=True)
    return centered / np.maximum(
        np.linalg.norm(centered, axis=-1, keepdims=True), 1e-6
    )


def spectral_angle_similarity(cube: np.ndarray, prototype: np.ndarray) -> np.ndarray:
    """Cosine form of Spectral Angle Mapper, mapped from [-1, 1] to [0, 1]."""
    similarity = _unit_shape(cube) @ _unit_shape(prototype)
    return np.clip((similarity + 1.0) * 0.5, 0.0, 1.0)


def ring_mask(mask: np.ndarray, scale: float = 1.75) -> np.ndarray:
    if scale <= 1:
        raise ValueError("ring scale must exceed one")
    points = cv2.findNonZero(mask.astype(np.uint8))
    if points is None:
        raise HyperspectralError("cannot make a ring around an empty mask")
    x, y, width, height = cv2.boundingRect(points)
    radius = max(2, round(max(width, height) * (scale - 1.0) / 2.0))
    # A dense morphology kernel becomes quadratic in object size (and was
    # prohibitively slow for large cards). Distance transform is linear in the
    # frame area and constructs the same outside-only local neighborhood.
    distance = cv2.distanceTransform(
        (~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_3
    )
    return (distance <= radius) & ~mask


@dataclass
class SpectralModel:
    target_prototype: np.ndarray
    background_prototype: np.ndarray
    classifier: object | None
    initial_target_score: float

    @classmethod
    def fit(
        cls,
        cube: np.ndarray,
        target_mask: np.ndarray,
        *,
        random_state: int = 2026,
        trees: int = 48,
        min_samples_leaf: int = 8,
    ) -> "SpectralModel":
        if cube.shape[:2] != target_mask.shape or not np.any(target_mask):
            raise HyperspectralError("initial target mask and cube do not align")
        background_mask = ring_mask(target_mask)
        target = cube[target_mask]
        background = cube[background_mask]
        target_prototype = np.median(target, axis=0).astype(np.float32)
        background_prototype = np.median(background, axis=0).astype(np.float32)

        classifier = None
        if trees:
            from sklearn.ensemble import ExtraTreesClassifier

            # The classifier is sequence-local and sees only frame-zero SAM mask
            # positives and its local ring. It is a nonlinear complement to SAM,
            # not a dataset-trained appearance tracker.
            max_background = min(len(background), max(4096, len(target) * 4))
            generator = np.random.default_rng(random_state)
            if len(background) > max_background:
                background = background[
                    generator.choice(len(background), max_background, replace=False)
                ]
            features = np.concatenate((target, background), axis=0)
            labels = np.concatenate(
                (
                    np.ones(len(target), dtype=np.uint8),
                    np.zeros(len(background), dtype=np.uint8),
                )
            )
            classifier = ExtraTreesClassifier(
                n_estimators=trees,
                min_samples_leaf=min_samples_leaf,
                max_features=None,
                class_weight="balanced",
                n_jobs=-1,
                random_state=random_state,
            ).fit(features, labels)

        model = cls(
            target_prototype=target_prototype,
            background_prototype=background_prototype,
            classifier=classifier,
            initial_target_score=0.0,
        )
        heatmap = model.heatmap(cube)
        model.initial_target_score = float(np.median(heatmap[target_mask]))
        return model

    def heatmap(
        self,
        cube: np.ndarray,
        *,
        sam_weight: float = 0.35,
        classifier_weight: float = 0.65,
        blur_sigma: float = 1.5,
    ) -> np.ndarray:
        target_similarity = spectral_angle_similarity(cube, self.target_prototype)
        background_similarity = spectral_angle_similarity(
            cube, self.background_prototype
        )
        # Local background contrast is explicit: resembling the target is useful
        # only insofar as the pixel resembles it more than the initial ring.
        sam_contrast = np.clip(
            0.5 + 0.5 * (target_similarity - background_similarity), 0.0, 1.0
        )
        if self.classifier is None:
            heatmap = sam_contrast
        else:
            probability = self.classifier.predict_proba(
                cube.reshape(-1, cube.shape[-1])
            )[:, 1].reshape(cube.shape[:2])
            heatmap = sam_weight * sam_contrast + classifier_weight * probability
        if blur_sigma > 0:
            heatmap = cv2.GaussianBlur(
                heatmap.astype(np.float32),
                (0, 0),
                sigmaX=blur_sigma,
                sigmaY=blur_sigma,
                borderType=cv2.BORDER_REFLECT101,
            )
        return np.clip(heatmap, 0.0, 1.0)

    def update(self, cube: np.ndarray, mask: np.ndarray, rate: float) -> None:
        """High-confidence prototype update; caller owns the memory veto."""
        if not 0 <= rate <= 1 or not np.any(mask):
            return
        observation = np.median(cube[mask], axis=0).astype(np.float32)
        self.target_prototype = (
            (1.0 - rate) * self.target_prototype + rate * observation
        )
