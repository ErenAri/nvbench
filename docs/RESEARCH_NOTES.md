# Research Notes

## Current research signal

Recent work supports the core NVBench direction: ODMR spectrum analysis is a real bottleneck in NV-diamond quantum sensing, especially when spectra are noisy, high-volume, or unsuitable for reliable nonlinear fitting.

Relevant source categories:

- Deep-learning ODMR spectrum analysis for direct parameter inference from NV-center spectra.
- Data-driven thermometry and sensing models for NV-diamond systems.
- Machine-learning current-density reconstruction from quantum diamond microscope magnetic images.
- Public Zenodo datasets containing ODMR, Ramsey, spin-echo, and widefield CW-ODMR data.

## Practical opportunity

The field already has papers using ML. The opportunity is not to claim novelty at the idea level. The opportunity is to build a clean open-source engineering artifact:

- dataset adapters
- physics-informed synthetic data
- classical fitting baselines
- ML baselines
- noise/latency/failure-rate benchmark suite
- dashboard and report export

## Initial data targets

### Single-NV ODMR dataset

Source: Zenodo record `14697917`.

Useful contents:

- ODMR data
- scan images
- Ramsey and spin-echo coherence data

Primary use:

- real-data adapter
- validation of synthetic-to-real transfer
- strain-related ODMR analysis experiments

### Widefield CW-ODMR FITS dataset

Source: Zenodo record `7233869`.

Useful contents:

- CW-ODMR magnetic imaging experiment data
- FITS image cubes

Primary use:

- widefield per-pixel ODMR inference
- future QDM-style imaging workflows

### FRET / chlorophyll ODMR dataset

Source: Zenodo record `18185936`.

Useful contents:

- fluorescence lifetime data
- raw ODMR data

Primary use:

- optional bio-sensing example after core benchmark is stable

## Technical hypothesis

Clean spectra:

- Lorentzian fitting should remain competitive and physically interpretable.

Noisy / low-SNR spectra:

- ML should reduce fitting failure rate and improve throughput.

High-throughput widefield spectra:

- ML should be more practical because it is parallelizable and avoids iterative optimization for every pixel.

Hybrid mode:

- ML can predict good initial parameters for classical fitting, combining speed and physical interpretability.

## Risk register

### Risk: synthetic-only benchmark overclaims

Mitigation:

- Always label synthetic vs real results.
- Add real-data adapters early.

### Risk: ML beats weak baseline only

Mitigation:

- Implement stronger fitting baselines.
- Include hybrid ML+fit mode.

### Risk: insufficient domain realism

Mitigation:

- Add realistic linewidth, contrast, baseline drift, overlapping dips, temperature shift, strain shift, and frequency-axis irregularity.

### Risk: weak GitHub credibility

Mitigation:

- Add CI, tests, reproducible reports, dataset cards, and benchmark tables.
