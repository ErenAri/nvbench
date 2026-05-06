from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from nvbench.data.loaders import load_synthetic_csv, pivot_spectra
from nvbench.fitting.lorentzian import fit_double_lorentzian
from nvbench.ml.train import train_splitting_regressor


def benchmark_fitting(frame: pd.DataFrame, limit: int | None = None) -> dict[str, Any]:
    freq, spectra, targets = pivot_spectra(frame)
    if limit is not None:
        spectra = spectra[:limit]
        targets = targets.iloc[:limit].reset_index(drop=True)

    predictions: list[float] = []
    success: list[bool] = []
    start = time.perf_counter()

    for spectrum in spectra:
        result = fit_double_lorentzian(freq, spectrum)
        predictions.append(result.splitting_mhz)
        success.append(result.success)

    elapsed = time.perf_counter() - start
    success_mask = np.asarray(success, dtype=bool)
    y_true = targets["splitting_mhz"].to_numpy(dtype=np.float64)
    y_pred = np.asarray(predictions, dtype=np.float64)

    if success_mask.any():
        mae = float(mean_absolute_error(y_true[success_mask], y_pred[success_mask]))
        rmse = float(np.sqrt(mean_squared_error(y_true[success_mask], y_pred[success_mask])))
    else:
        mae = float("nan")
        rmse = float("nan")

    return {
        "method": "double_lorentzian_curve_fit",
        "samples": int(len(spectra)),
        "success_rate": float(success_mask.mean()),
        "mae_splitting_mhz": mae,
        "rmse_splitting_mhz": rmse,
        "total_seconds": float(elapsed),
        "milliseconds_per_spectrum": float(elapsed * 1000.0 / max(len(spectra), 1)),
    }


def benchmark_ml(frame: pd.DataFrame) -> dict[str, Any]:
    freq, spectra, targets = pivot_spectra(frame)
    start = time.perf_counter()
    _, report = train_splitting_regressor(
        frequency_mhz=freq,
        spectra=spectra,
        splitting_mhz=targets["splitting_mhz"].to_numpy(dtype=np.float64),
    )
    elapsed = time.perf_counter() - start
    return {
        "method": "random_forest_feature_regressor",
        "mae_splitting_mhz": report.mae,
        "rmse_splitting_mhz": report.rmse,
        "train_size": report.train_size,
        "test_size": report.test_size,
        "total_seconds": float(elapsed),
    }


def run_fitting_benchmark(input_path: str | Path, output_path: str | Path, limit: int | None = None) -> Path:
    frame = load_synthetic_csv(input_path)
    report = benchmark_fitting(frame, limit=limit)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output


def run_ml_benchmark(input_path: str | Path, output_path: str | Path) -> Path:
    frame = load_synthetic_csv(input_path)
    report = benchmark_ml(frame)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output
