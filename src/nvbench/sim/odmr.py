from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class ODMRParameters:
    center_mhz: float
    splitting_mhz: float
    linewidth_mhz: float
    contrast: float
    baseline: float
    noise_std: float


@dataclass(frozen=True)
class ODMRSpectrum:
    frequency_mhz: NDArray[np.float64]
    intensity: NDArray[np.float64]
    parameters: ODMRParameters


def lorentzian_dip(
    frequency_mhz: NDArray[np.float64],
    center_mhz: float,
    linewidth_mhz: float,
    contrast: float,
) -> NDArray[np.float64]:
    gamma = linewidth_mhz / 2.0
    return contrast * (gamma**2 / ((frequency_mhz - center_mhz) ** 2 + gamma**2))


def generate_single_spectrum(
    frequency_mhz: NDArray[np.float64],
    center_mhz: float = 2870.0,
    splitting_mhz: float = 6.0,
    linewidth_mhz: float = 5.0,
    contrast: float = 0.04,
    baseline: float = 1.0,
    noise_std: float = 0.003,
    rng: np.random.Generator | None = None,
) -> ODMRSpectrum:
    generator = rng or np.random.default_rng()
    left_center = center_mhz - splitting_mhz / 2.0
    right_center = center_mhz + splitting_mhz / 2.0
    clean = baseline
    clean -= lorentzian_dip(frequency_mhz, left_center, linewidth_mhz, contrast)
    clean -= lorentzian_dip(frequency_mhz, right_center, linewidth_mhz, contrast)
    noisy = clean + generator.normal(0.0, noise_std, size=frequency_mhz.shape)
    parameters = ODMRParameters(
        center_mhz=center_mhz,
        splitting_mhz=splitting_mhz,
        linewidth_mhz=linewidth_mhz,
        contrast=contrast,
        baseline=baseline,
        noise_std=noise_std,
    )
    return ODMRSpectrum(frequency_mhz=frequency_mhz, intensity=noisy, parameters=parameters)


def generate_dataset(
    samples: int,
    points: int = 401,
    frequency_min_mhz: float = 2830.0,
    frequency_max_mhz: float = 2910.0,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frequency = np.linspace(frequency_min_mhz, frequency_max_mhz, points, dtype=np.float64)
    rows: list[dict[str, float | int | str]] = []

    for sample_id in range(samples):
        center = rng.normal(2870.0, 1.5)
        splitting = rng.uniform(0.5, 24.0)
        linewidth = rng.uniform(2.0, 12.0)
        contrast = rng.uniform(0.01, 0.09)
        baseline = rng.normal(1.0, 0.01)
        noise = rng.uniform(0.0005, 0.015)
        spectrum = generate_single_spectrum(
            frequency_mhz=frequency,
            center_mhz=center,
            splitting_mhz=splitting,
            linewidth_mhz=linewidth,
            contrast=contrast,
            baseline=baseline,
            noise_std=noise,
            rng=rng,
        )
        for freq, intensity in zip(spectrum.frequency_mhz, spectrum.intensity, strict=True):
            rows.append(
                {
                    "sample_id": sample_id,
                    "frequency_mhz": float(freq),
                    "intensity": float(intensity),
                    "center_mhz": center,
                    "splitting_mhz": splitting,
                    "linewidth_mhz": linewidth,
                    "contrast": contrast,
                    "baseline": baseline,
                    "noise_std": noise,
                }
            )

    return pd.DataFrame(rows)


def save_synthetic_dataset(path: str | Path, samples: int, points: int = 401, seed: int = 42) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_dataset(samples=samples, points=points, seed=seed)
    frame.to_csv(output, index=False)
    return output
