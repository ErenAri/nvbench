from nvbench.benchmarks.evaluate import benchmark_fitting, benchmark_ml
from nvbench.sim.odmr import generate_dataset


def main() -> None:
    frame = generate_dataset(samples=100, points=301, seed=7)
    fitting_report = benchmark_fitting(frame, limit=20)
    ml_report = benchmark_ml(frame)
    print(fitting_report)
    print(ml_report)


if __name__ == "__main__":
    main()
