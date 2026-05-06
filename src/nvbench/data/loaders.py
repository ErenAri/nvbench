from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class SpectrumBatch:
    sample_ids: NDArray[np.int64]
    frequency_mhz: NDArray[np.float64]
    intensity: NDArray[np.float64]
    targets: pd.DataFrame


def load_synthetic_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"sample_id", "frequency_mhz", "intensity", "splitting_mhz"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return frame


def pivot_spectra(frame: pd.DataFrame) -> tuple[NDArray[np.float64], NDArray[np.float64], pd.DataFrame]:
    ordered = frame.sort_values(["sample_id", "frequency_mhz"])
    frequencies = ordered.groupby("sample_id")["frequency_mhz"].apply(lambda value: value.to_numpy())
    intensities = ordered.groupby("sample_id")["intensity"].apply(lambda value: value.to_numpy())
    x = np.stack(intensities.to_list()).astype(np.float64)
    freq = frequencies.iloc[0].astype(np.float64)
    targets = ordered.groupby("sample_id").first()[
        ["center_mhz", "splitting_mhz", "linewidth_mhz", "contrast", "baseline", "noise_std"]
    ]
    return freq, x, targets.reset_index()


def load_fits_cube(path: str | Path) -> NDArray[np.float64]:
    try:
        from astropy.io import fits
    except ImportError as exc:
        raise ImportError("Install nvbench[fits] to read FITS data") from exc

    with fits.open(path) as hdul:
        return np.asarray(hdul[0].data, dtype=np.float64)
