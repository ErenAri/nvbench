from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from nvbench.benchmarks.evaluate import run_fitting_benchmark, run_ml_benchmark
from nvbench.sim.odmr import save_synthetic_dataset

app = typer.Typer(help="NVBench CLI")
console = Console()


@app.command()
def generate(
    samples: int = typer.Option(1000, min=1),
    points: int = typer.Option(401, min=51),
    output: Path = typer.Option(Path("data/processed/synthetic_odmr.csv")),
    seed: int = typer.Option(42),
) -> None:
    path = save_synthetic_dataset(output, samples=samples, points=points, seed=seed)
    console.print(f"Synthetic ODMR dataset written to {path}")


@app.command("fit-benchmark")
def fit_benchmark(
    input: Path = typer.Option(Path("data/processed/synthetic_odmr.csv")),
    output: Path = typer.Option(Path("reports/fitting_report.json")),
    limit: int | None = typer.Option(None),
) -> None:
    path = run_fitting_benchmark(input, output, limit=limit)
    console.print(f"Fitting benchmark report written to {path}")


@app.command("ml-benchmark")
def ml_benchmark(
    input: Path = typer.Option(Path("data/processed/synthetic_odmr.csv")),
    output: Path = typer.Option(Path("reports/ml_report.json")),
) -> None:
    path = run_ml_benchmark(input, output)
    console.print(f"ML benchmark report written to {path}")


if __name__ == "__main__":
    app()
