# NVBench Project Plan

## Objective

Build an open benchmark and inference toolkit for NV-diamond ODMR analysis. The project should help researchers compare classical curve fitting against machine-learning inference under realistic operating constraints: noise, drift, poor initialization, and high-throughput widefield acquisition.

## Strategic positioning

NVBench is not positioned as a replacement for physics-based fitting. The correct positioning is narrower and stronger:

> Classical fitting remains the reference method for clean, well-conditioned spectra. ML inference becomes valuable when ODMR analysis must be robust, low-latency, parallelizable, and tolerant to low signal-to-noise regimes.

## Phase 1: Core ODMR benchmark

Deliverables:

- Synthetic ODMR spectrum generator
- Double-Lorentzian fitting baseline
- Feature-based ML baseline
- CLI commands for data generation and benchmark reports
- Unit tests and CI

Success criteria:

- Reproducible synthetic dataset generation
- Reported MAE/RMSE for resonance splitting
- Reported fitting success rate
- Reported latency per spectrum
- Clear README and install path

## Phase 2: Research-grade ML baseline

Deliverables:

- 1D-CNN model for spectrum-to-parameter inference
- Probabilistic or ensemble-based uncertainty estimation
- Noise sweep benchmark
- Hybrid inference mode: ML initialization followed by nonlinear fitting
- Model export and reproducible training configuration

Success criteria:

- Better low-SNR robustness than naive fitting
- Lower inference latency than iterative fitting after training
- Calibration metrics for uncertainty quality
- Benchmark table comparing fitting, RandomForest, 1D-CNN, and hybrid ML+fit

## Phase 3: Real-data adapters

Deliverables:

- Zenodo ODMR dataset adapter
- Widefield FITS cube loader
- Per-pixel ODMR inference path
- Dataset cards with source, license, and expected schema

Success criteria:

- At least one real single-NV ODMR dataset can be loaded and analyzed
- At least one widefield dataset can be loaded for per-pixel inference experiments
- Synthetic-to-real gap is explicitly reported

## Phase 4: Dashboard and reproducible reports

Deliverables:

- Streamlit dashboard
- Spectrum upload and visualization
- Fitting vs ML prediction comparison
- Latency and robustness report export
- Example notebook or script for real data

Success criteria:

- A user can upload a CSV spectrum and receive fitted/ML-estimated parameters
- Reports are exportable as JSON and Markdown
- Demo can run locally without special hardware

## Phase 5: QDM current-density reconstruction extension

Deliverables:

- Synthetic current map generator
- Biot-Savart forward model for magnetic field maps
- U-Net/CNN inverse model
- Analytic inversion baseline
- Noise and standoff-distance robustness benchmark

Success criteria:

- Reconstruct current-density maps from noisy vector magnetic images
- Compare ML inversion against analytic reconstruction
- Explicitly separate synthetic-only claims from real-data claims

## Non-goals for early versions

- No claim of universal superiority over fitting
- No medical/diagnostic claims
- No hardware-control layer
- No closed-source/proprietary semiconductor failure-analysis data dependency
- No quantum-computing qubit-control claims unless supported by actual experiments

## Repository standards

- Every benchmark should be reproducible from a CLI command
- Every result should include dataset seed/configuration
- Every model should have a baseline comparator
- Every scientific claim should identify whether it comes from synthetic or real data
- Every future real dataset should include a dataset card
