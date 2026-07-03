import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Compare prompt versions for a model and benchmark.")
    parser.add_argument("--leaderboard", default="results/leaderboard.csv")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--model-name", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    leaderboard_path = Path(args.leaderboard)

    if not leaderboard_path.exists():
        raise FileNotFoundError(f"Leaderboard not found: {leaderboard_path}")

    leaderboard_df = pd.read_csv(leaderboard_path).fillna("")

    filtered = leaderboard_df[
        (leaderboard_df["benchmark"] == args.benchmark)
        & (leaderboard_df["model_name"] == args.model_name)
        & (leaderboard_df["prompt_path"] != "")
    ]

    rows = []

    for _, run in filtered.iterrows():
        results_path = Path(run["output_dir"]) / "results.csv"

        if not results_path.exists():
            continue

        results_df = pd.read_csv(results_path).fillna("")

        if "failure_type" not in results_df.columns:
            continue

        counts = results_df["failure_type"].value_counts().to_dict()

        rows.append(
            {
                "timestamp": run["timestamp"],
                "run_name": run["run_name"],
                "prompt_path": run["prompt_path"],
                "accuracy": run["accuracy"],
                "correct": counts.get("correct", 0),
                "no_numeric_answer": counts.get("no_numeric_answer", 0),
                "format_violation": counts.get("format_violation", 0),
                "wrong_numeric_answer": counts.get("wrong_numeric_answer", 0),
            }
        )

    if not rows:
        print("No prompt comparison rows found.")
        return

    df = pd.DataFrame(rows)

    print("=" * 120)
    print("Prompt Comparison")
    print("=" * 120)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
