from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class NoiseConfig:
    gaussian_std: float = 0.0
    baseline_slope: float = 0.0
    baseline_curvature: float = 0.0
    frequency_jitter_std_mhz: float = 0.0
    thermal_shift_mhz: float = 0.0
    contrast_scale: float = 1.0
    broadening_sigma_mhz: float = 0.0


def add_gaussian_noise(
    intensity: NDArray[np.float64],
    std: float,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    if std <= 0.0:
        return intensity.astype(np.float64, copy=True)
    generator = rng or np.random.default_rng()
    return intensity.astype(np.float64, copy=True) + generator.normal(0.0, std, size=intensity.shape)


def add_baseline_drift(
    frequency_mhz: NDArray[np.float64],
    intensity: NDArray[np.float64],
    slope: float = 0.0,
    curvature: float = 0.0,
) -> NDArray[np.float64]:
    if slope == 0.0 and curvature == 0.0:
        return intensity.astype(np.float64, copy=True)

    span = float(frequency_mhz.max() - frequency_mhz.min())
    if span <= 0.0:
        raise ValueError("frequency_mhz must span a non-zero range")

    normalized = (frequency_mhz - frequency_mhz.mean()) / span
    curved = normalized**2 - float(np.mean(normalized**2))
    drift = slope * normalized + curvature * curved
    return intensity.astype(np.float64, copy=True) + drift


def add_frequency_jitter(
    frequency_mhz: NDArray[np.float64],
    intensity: NDArray[np.float64],
    std_mhz: float,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    if std_mhz <= 0.0:
        return intensity.astype(np.float64, copy=True)

    generator = rng or np.random.default_rng()
    jitter = generator.normal(0.0, std_mhz, size=frequency_mhz.shape)
    sampled_at = frequency_mhz + jitter
    return np.interp(
        sampled_at,
        frequency_mhz,
        intensity,
        left=float(intensity[0]),
        right=float(intensity[-1]),
    ).astype(np.float64)


def add_thermal_shift(
    frequency_mhz: NDArray[np.float64],
    intensity: NDArray[np.float64],
    shift_mhz: float,
) -> NDArray[np.float64]:
    if shift_mhz == 0.0:
        return intensity.astype(np.float64, copy=True)

    return np.interp(
        frequency_mhz - shift_mhz,
        frequency_mhz,
        intensity,
        left=float(intensity[0]),
        right=float(intensity[-1]),
    ).astype(np.float64)


def add_contrast_degradation(
    intensity: NDArray[np.float64],
    baseline: float,
    scale: float,
) -> NDArray[np.float64]:
    if scale < 0.0:
        raise ValueError("contrast scale must be non-negative")
    if scale == 1.0:
        return intensity.astype(np.float64, copy=True)
    return baseline - (baseline - intensity.astype(np.float64, copy=True)) * scale


def add_power_broadening(
    frequency_mhz: NDArray[np.float64],
    intensity: NDArray[np.float64],
    sigma_mhz: float,
) -> NDArray[np.float64]:
    if sigma_mhz <= 0.0:
        return intensity.astype(np.float64, copy=True)

    step = float(np.median(np.diff(frequency_mhz)))
    if step <= 0.0:
        raise ValueError("frequency_mhz must be strictly increasing")

    sigma_points = max(sigma_mhz / step, 1e-6)
    radius = max(int(np.ceil(4.0 * sigma_points)), 1)
    grid = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (grid / sigma_points) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(intensity.astype(np.float64), pad_width=radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float64)


def apply_noise_model(
    frequency_mhz: NDArray[np.float64],
    intensity: NDArray[np.float64],
    config: NoiseConfig,
    baseline: float = 1.0,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    generator = rng or np.random.default_rng()
    degraded = add_contrast_degradation(intensity, baseline=baseline, scale=config.contrast_scale)
    broadened = add_power_broadening(frequency_mhz, degraded, sigma_mhz=config.broadening_sigma_mhz)
    shifted = add_thermal_shift(frequency_mhz, broadened, shift_mhz=config.thermal_shift_mhz)
    jittered = add_frequency_jitter(
        frequency_mhz,
        shifted,
        std_mhz=config.frequency_jitter_std_mhz,
        rng=generator,
    )
    drifted = add_baseline_drift(
        frequency_mhz,
        jittered,
        slope=config.baseline_slope,
        curvature=config.baseline_curvature,
    )
    return add_gaussian_noise(drifted, std=config.gaussian_std, rng=generator)
