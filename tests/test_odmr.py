import numpy as np

from nvbench.fitting.lorentzian import fit_double_lorentzian
from nvbench.sim.odmr import generate_dataset, generate_single_spectrum


def test_generate_dataset_has_expected_columns() -> None:
    frame = generate_dataset(samples=3, points=101, seed=1)
    assert len(frame) == 303
    assert "frequency_mhz" in frame.columns
    assert "splitting_mhz" in frame.columns


def test_fit_recovers_clean_splitting() -> None:
    frequency = np.linspace(2830.0, 2910.0, 401)
    spectrum = generate_single_spectrum(
        frequency_mhz=frequency,
        splitting_mhz=10.0,
        linewidth_mhz=5.0,
        contrast=0.05,
        noise_std=0.0,
    )
    result = fit_double_lorentzian(frequency, spectrum.intensity)
    assert result.success
    assert abs(result.splitting_mhz - 10.0) < 0.5
