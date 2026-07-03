import argparse
from pathlib import Path

from inspect_failures import main as inspect_main
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect failures from the latest benchmark run.")
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Benchmark name, e.g. gsm8k.",
    )
    parser.add_argument(
        "--results-base-dir",
        default="results",
        help="Base results directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of failed examples to print.",
    )
    return parser.parse_args()


def find_latest_results_file(results_base_dir: str, benchmark: str) -> Path:
    benchmark_dir = Path(results_base_dir) / benchmark

    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Benchmark results directory not found: {benchmark_dir}")

    run_dirs = [p for p in benchmark_dir.iterdir() if p.is_dir()]

    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under: {benchmark_dir}")

    latest_run_dir = sorted(run_dirs, key=lambda p: p.name)[-1]
    results_file = latest_run_dir / "results.csv"

    if not results_file.exists():
        raise FileNotFoundError(f"results.csv not found in latest run directory: {latest_run_dir}")

    return results_file


def main():
    args = parse_args()

    results_file = find_latest_results_file(
        results_base_dir=args.results_base_dir,
        benchmark=args.benchmark,
    )

    print(f"Inspecting latest results file: {results_file}")

    sys.argv = [
        "inspect_failures.py",
        "--results",
        str(results_file),
        "--limit",
        str(args.limit),
    ]

    inspect_main()


if __name__ == "__main__":
    main()
