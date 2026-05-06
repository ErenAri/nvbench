from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class FitResult:
    center_mhz: float
    splitting_mhz: float
    linewidth_mhz: float
    contrast: float
    baseline: float
    success: bool
    error: str | None = None


def double_lorentzian_model(
    frequency_mhz: NDArray[np.float64],
    center_mhz: float,
    splitting_mhz: float,
    linewidth_mhz: float,
    contrast: float,
    baseline: float,
) -> NDArray[np.float64]:
    gamma = linewidth_mhz / 2.0
    left = center_mhz - splitting_mhz / 2.0
    right = center_mhz + splitting_mhz / 2.0
    left_dip = contrast * (gamma**2 / ((frequency_mhz - left) ** 2 + gamma**2))
    right_dip = contrast * (gamma**2 / ((frequency_mhz - right) ** 2 + gamma**2))
    return baseline - left_dip - right_dip


def fit_double_lorentzian(
    frequency_mhz: NDArray[np.float64],
    intensity: NDArray[np.float64],
) -> FitResult:
    baseline_guess = float(np.percentile(intensity, 95))
    center_guess = float(frequency_mhz[np.argmin(intensity)])
    initial = [center_guess, 6.0, 5.0, 0.03, baseline_guess]
    lower = [frequency_mhz.min(), 0.0, 0.2, 0.0, 0.5]
    upper = [frequency_mhz.max(), 60.0, 50.0, 0.3, 1.5]

    try:
        params, _ = curve_fit(
            double_lorentzian_model,
            frequency_mhz,
            intensity,
            p0=initial,
            bounds=(lower, upper),
            maxfev=10_000,
        )
        return FitResult(
            center_mhz=float(params[0]),
            splitting_mhz=float(params[1]),
            linewidth_mhz=float(params[2]),
            contrast=float(params[3]),
            baseline=float(params[4]),
            success=True,
        )
    except Exception as exc:
        return FitResult(
            center_mhz=float("nan"),
            splitting_mhz=float("nan"),
            linewidth_mhz=float("nan"),
            contrast=float("nan"),
            baseline=float("nan"),
            success=False,
            error=str(exc),
        )
