import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Compare failure types across evaluation runs.")
    parser.add_argument(
        "--leaderboard",
        default="results/leaderboard.csv",
        help="Path to leaderboard CSV.",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Optional benchmark filter, e.g. gsm8k.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    leaderboard_path = Path(args.leaderboard)

    if not leaderboard_path.exists():
        raise FileNotFoundError(f"Leaderboard not found: {leaderboard_path}")

    leaderboard_df = pd.read_csv(leaderboard_path).fillna("")

    if args.benchmark:
        leaderboard_df = leaderboard_df[leaderboard_df["benchmark"] == args.benchmark]

    rows = []

    for _, run in leaderboard_df.iterrows():
        output_dir = Path(run["output_dir"])
        results_path = output_dir / "results.csv"

        if not results_path.exists():
            continue

        results_df = pd.read_csv(results_path).fillna("")

        if "failure_type" not in results_df.columns:
            continue

        counts = results_df["failure_type"].value_counts().to_dict()

        row = {
            "timestamp": run.get("timestamp", ""),
            "run_name": run.get("run_name", ""),
            "model_name": run.get("model_name", ""),
            "benchmark": run.get("benchmark", ""),
            "limit": run.get("limit", ""),
            "accuracy": run.get("accuracy", ""),
            "prompt_path": run.get("prompt_path", ""),
            "correct": counts.get("correct", 0),
            "no_numeric_answer": counts.get("no_numeric_answer", 0),
            "format_violation": counts.get("format_violation", 0),
            "wrong_numeric_answer": counts.get("wrong_numeric_answer", 0),
        }

        rows.append(row)

    if not rows:
        print("No runs with failure_type data found.")
        return

    comparison_df = pd.DataFrame(rows)

    print("=" * 140)
    print("Failure Type Comparison")
    print("=" * 140)
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
