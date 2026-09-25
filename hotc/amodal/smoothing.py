"""RTS smoothing for the released empty-mask fallback."""

from __future__ import annotations

import numpy as np


def _constant_velocity_noise(acceleration_std: np.ndarray) -> np.ndarray:
    noise = np.zeros((8, 8), dtype=np.float64)
    for axis, std in enumerate(acceleration_std):
        variance = float(std) ** 2
        noise[axis, axis] = variance * 0.25
        noise[axis, axis + 4] = variance * 0.5
        noise[axis + 4, axis] = variance * 0.5
        noise[axis + 4, axis + 4] = variance
    return noise


def _rts_smooth(
    observations: np.ndarray,
    reliable: np.ndarray,
    *,
    center_acceleration_std: float = 2.0,
    size_acceleration_std: float = 0.03,
    center_measurement_ratio: float = 0.03,
    size_measurement_std: float = 0.06,
) -> np.ndarray:
    """Run a constant-velocity Kalman filter and RTS backward pass."""
    count = len(observations)
    reliable = np.asarray(reliable, dtype=bool)
    if reliable.shape == (count,):
        reliable = np.broadcast_to(reliable[:, None], (count, 4)).copy()
    if reliable.shape != (count, 4):
        raise ValueError("reliable must have shape (frames,) or (frames, 4)")
    transition = np.eye(8, dtype=np.float64)
    transition[:4, 4:] = np.eye(4, dtype=np.float64)
    observation_model = np.zeros((4, 8), dtype=np.float64)
    observation_model[:, :4] = np.eye(4, dtype=np.float64)
    process_noise = _constant_velocity_noise(
        np.asarray(
            [
                center_acceleration_std,
                center_acceleration_std,
                size_acceleration_std,
                size_acceleration_std,
            ]
        )
    )

    filtered_states = np.empty((count, 8), dtype=np.float64)
    filtered_covariances = np.empty((count, 8, 8), dtype=np.float64)
    predicted_states = np.empty_like(filtered_states)
    predicted_covariances = np.empty_like(filtered_covariances)
    state = np.concatenate((observations[0], np.zeros(4, dtype=np.float64)))
    covariance = np.diag([4.0, 4.0, 0.01, 0.01, 25.0, 25.0, 0.01, 0.01])
    identity = np.eye(8, dtype=np.float64)

    for index in range(count):
        if index:
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process_noise
        predicted_states[index] = state
        predicted_covariances[index] = covariance
        component_indices = np.flatnonzero(reliable[index])
        if len(component_indices):
            width = float(np.exp(observations[index, 2]))
            height = float(np.exp(observations[index, 3]))
            full_measurement_noise = np.diag(
                [
                    max(2.0, width * center_measurement_ratio) ** 2,
                    max(2.0, height * center_measurement_ratio) ** 2,
                    size_measurement_std**2,
                    size_measurement_std**2,
                ]
            )
            current_observation_model = observation_model[component_indices]
            measurement_noise = full_measurement_noise[
                np.ix_(component_indices, component_indices)
            ]
            innovation_covariance = (
                current_observation_model
                @ covariance
                @ current_observation_model.T
                + measurement_noise
            )
            gain = (
                covariance
                @ current_observation_model.T
                @ np.linalg.pinv(innovation_covariance)
            )
            innovation = (
                observations[index, component_indices]
                - current_observation_model @ state
            )
            state = state + gain @ innovation
            covariance = (
                (identity - gain @ current_observation_model)
                @ covariance
                @ (identity - gain @ current_observation_model).T
                + gain @ measurement_noise @ gain.T
            )
        filtered_states[index] = state
        filtered_covariances[index] = covariance

    smoothed_states = filtered_states.copy()
    smoothed_covariances = filtered_covariances.copy()
    for index in range(count - 2, -1, -1):
        smoother_gain = (
            filtered_covariances[index]
            @ transition.T
            @ np.linalg.pinv(predicted_covariances[index + 1])
        )
        smoothed_states[index] += smoother_gain @ (
            smoothed_states[index + 1] - predicted_states[index + 1]
        )
        smoothed_covariances[index] += smoother_gain @ (
            smoothed_covariances[index + 1]
            - predicted_covariances[index + 1]
        ) @ smoother_gain.T
    return smoothed_states[:, :4]
