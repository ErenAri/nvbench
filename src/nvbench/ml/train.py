from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from nvbench.ml.features import spectrum_features


@dataclass(frozen=True)
class RegressionReport:
    mae: float
    rmse: float
    train_size: int
    test_size: int


def train_splitting_regressor(
    frequency_mhz: NDArray[np.float64],
    spectra: NDArray[np.float64],
    splitting_mhz: NDArray[np.float64],
    seed: int = 42,
) -> tuple[RandomForestRegressor, RegressionReport]:
    features = spectrum_features(frequency_mhz, spectra)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        splitting_mhz,
        test_size=0.2,
        random_state=seed,
    )
    model = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    report = RegressionReport(
        mae=float(mean_absolute_error(y_test, prediction)),
        rmse=float(mean_squared_error(y_test, prediction, squared=False)),
        train_size=int(len(y_train)),
        test_size=int(len(y_test)),
    )
    return model, report
