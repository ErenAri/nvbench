# NVBench

Open benchmark and ML toolkit for NV-diamond ODMR magnetometry analysis.

NVBench targets a practical bottleneck in NV-center quantum sensing: extracting physical parameters from ODMR spectra under noise, drift, and high-throughput constraints. The initial release includes synthetic ODMR generation, classical Lorentzian fitting, baseline ML regression, and benchmark utilities.

## Project scope

```text
ODMR spectrum
→ preprocessing
→ Lorentzian fitting baseline
→ ML inference baseline
→ robustness benchmark
→ reportable metrics
```

## Current MVP features

- Synthetic ODMR spectrum generation
- Single and multi-dip Lorentzian models
- Nonlinear curve-fitting baseline
- Feature-based ML baseline using scikit-learn
- Noise robustness benchmark
- CLI for data generation and evaluation
- Dataset loader stubs for CSV and FITS-based workflows

## Why this project matters

Classical ODMR fitting is strong on clean spectra, but it can become slow and brittle under low signal-to-noise ratio, distorted resonance dips, high-throughput widefield settings, or poor initialization. NVBench is designed to measure when classical fitting is enough and when ML inference becomes more practical.

## Installation

```bash
git clone https://github.com/ErenAri/nvbench.git
cd nvbench
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On Windows PowerShell:

```powershell
git clone https://github.com/ErenAri/nvbench.git
cd nvbench
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Quick start

Generate synthetic ODMR data:

```bash
nvbench generate --samples 1000 --output data/processed/synthetic_odmr.csv
```

Benchmark Lorentzian fitting:

```bash
nvbench fit-benchmark --input data/processed/synthetic_odmr.csv --output reports/fitting_report.json
```

Run the example script:

```bash
python examples/quickstart.py
```

## Repository layout

```text
src/nvbench/
  sim/          Synthetic ODMR generators
  fitting/      Lorentzian fitting baselines
  ml/           ML feature extraction and models
  benchmarks/   Evaluation metrics and benchmark runners
  data/         CSV and FITS data loading utilities
examples/       Minimal usage examples
scripts/        Reproducible scripts
tests/          Unit tests
```

## Data sources to support next

- Zenodo ODMR datasets from NV-center experiments
- Widefield CW-ODMR FITS datasets
- Synthetic ODMR generated from Hamiltonian/spectrum simulators

## Near-term roadmap

- 1D-CNN spectrum regression baseline
- Uncertainty estimation head
- Synthetic-to-real evaluation protocol
- Streamlit dashboard
- Widefield per-pixel ODMR map inference
- Current-density reconstruction extension

## Scientific positioning

NVBench should not claim that ML universally replaces fitting. The correct claim is narrower:

> Classical fitting remains the reference method on clean and well-conditioned spectra. ML becomes useful when the goal is robust, high-throughput, low-latency ODMR inference under noise and operational constraints.

## License

MIT
