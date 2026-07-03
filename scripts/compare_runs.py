import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Compare evaluation runs from leaderboard.csv.")
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
    path = Path(args.leaderboard)

    if not path.exists():
        raise FileNotFoundError(f"Leaderboard not found: {path}")

    df = pd.read_csv(path).fillna("")

    if args.benchmark:
        df = df[df["benchmark"] == args.benchmark]

    columns = [
        "timestamp",
        "run_name",
        "model_name",
        "benchmark",
        "limit",
        "accuracy",
        "max_new_tokens",
        "temperature",
        "top_p",
        "prompt_path",
        "output_dir",
    ]

    existing_columns = [c for c in columns if c in df.columns]
    df = df[existing_columns]

    print("=" * 120)
    print("Evaluation Run Comparison")
    print("=" * 120)

    if df.empty:
        print("No matching runs found.")
        return

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
