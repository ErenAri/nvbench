from pathlib import Path

from nvbench.sim.odmr import save_synthetic_dataset


if __name__ == "__main__":
    save_synthetic_dataset(Path("data/processed/synthetic_odmr.csv"), samples=5000, points=401)
