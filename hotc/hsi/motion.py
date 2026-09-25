"""Static-scene affine motion estimation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AffineMotion:
    translation: float
    scale: float
    rotation_degrees: float
    inlier_fraction: float
    points: int


def _mask_for_features(
    mask: np.ndarray | None, shape: tuple[int, int], *, exclude: bool
) -> np.ndarray:
    result = np.full(shape, 255, dtype=np.uint8)
    if mask is None:
        return result
    value = np.asarray(mask, dtype=bool)
    if value.shape != shape:
        raise ValueError(f"feature mask {value.shape} does not match image {shape}")
    expanded = cv2.dilate(value.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    if exclude:
        result[expanded] = 0
    else:
        result.fill(0)
        result[expanded] = 255
    return result


def estimate_affine_motion(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    exclude_mask: bool = False,
    maximum_corners: int = 600,
) -> AffineMotion | None:
    previous = np.asarray(previous_gray, dtype=np.uint8)
    current = np.asarray(current_gray, dtype=np.uint8)
    if previous.shape != current.shape or previous.ndim != 2:
        raise ValueError("optical-flow images must be aligned grayscale arrays")
    feature_mask = _mask_for_features(mask, previous.shape, exclude=exclude_mask)
    points = cv2.goodFeaturesToTrack(
        previous,
        maxCorners=maximum_corners,
        qualityLevel=0.005,
        minDistance=3,
        mask=feature_mask,
        blockSize=5,
    )
    if points is None or len(points) < 4:
        return None
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        previous,
        current,
        points,
        None,
        winSize=(21, 21),
        maxLevel=3,
    )
    valid = np.asarray(status).reshape(-1).astype(bool)
    source = points.reshape(-1, 2)[valid]
    target = tracked.reshape(-1, 2)[valid]
    if len(source) < 4:
        return None
    transform, inliers = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=1.0,
        maxIters=3000,
        confidence=0.995,
    )
    if transform is None or inliers is None:
        return None
    scale = float(np.hypot(transform[0, 0], transform[0, 1]))
    rotation = float(np.degrees(np.arctan2(transform[1, 0], transform[0, 0])))
    return AffineMotion(
        translation=float(np.linalg.norm(transform[:, 2])),
        scale=scale,
        rotation_degrees=rotation,
        inlier_fraction=float(np.asarray(inliers).mean()),
        points=len(source),
    )
