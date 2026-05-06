from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def normalize_spectra(x: NDArray[np.float64]) -> NDArray[np.float64]:
    min_values = x.min(axis=1, keepdims=True)
    max_values = x.max(axis=1, keepdims=True)
    scale = np.maximum(max_values - min_values, 1e-12)
    return (x - min_values) / scale


def spectrum_features(
    frequency_mhz: NDArray[np.float64],
    spectra: NDArray[np.float64],
) -> NDArray[np.float64]:
    norm = normalize_spectra(spectra)
    idx_min = np.argmin(norm, axis=1)
    min_freq = frequency_mhz[idx_min]
    min_value = norm[np.arange(norm.shape[0]), idx_min]
    mean_value = norm.mean(axis=1)
    std_value = norm.std(axis=1)
    q05 = np.quantile(norm, 0.05, axis=1)
    q50 = np.quantile(norm, 0.50, axis=1)
    q95 = np.quantile(norm, 0.95, axis=1)
    return np.column_stack([min_freq, min_value, mean_value, std_value, q05, q50, q95])
