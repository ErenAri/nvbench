# Dataset Targets

## Zenodo: Nitrogen-Vacancy Centers in Epitaxial Laterally Overgrown Diamond

Record: `14697917`

Use:

- single NV ODMR loading
- Ramsey/spin-echo future extension
- synthetic-to-real validation

Expected adapter:

```text
nvbench datasets download-single-nv
nvbench datasets prepare-single-nv
```

## Zenodo: Wide-field magnetometry using randomly oriented micro-diamonds

Record: `7233869`

Use:

- FITS cube loading
- widefield per-pixel ODMR inference
- imaging benchmark

Expected adapter:

```text
nvbench datasets download-widefield
nvbench datasets inspect-fits
```

## Zenodo: FRET between NV centers and chlorophyll molecules

Record: `18185936`

Use:

- optional bio-sensing ODMR example
- multimodal sensing extension

Expected adapter:

```text
nvbench datasets download-fret
```

## Local CSV schema

Minimum required columns:

```text
sample_id
frequency_mhz
intensity
splitting_mhz
```

Recommended target columns:

```text
center_mhz
linewidth_mhz
contrast
baseline
noise_std
temperature_k
strain_mhz
source
```
