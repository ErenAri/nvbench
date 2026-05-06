# Contributing

NVBench is early-stage. Contributions should improve reproducibility, scientific correctness, or benchmark coverage.

## Useful contribution areas

- Real ODMR dataset loaders
- Better physical simulators
- Robust fitting baselines
- CNN and Transformer spectrum models
- Uncertainty calibration
- Widefield FITS workflows
- Benchmark reports

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest -q
```
