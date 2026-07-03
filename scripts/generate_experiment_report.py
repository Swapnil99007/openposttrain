import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Generate markdown experiment report.")
    parser.add_argument("--leaderboard", default="results/leaderboard.csv")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", default="reports/gsm8k_report.md")
    return parser.parse_args()


def main():
    args = parse_args()

    leaderboard_path = Path(args.leaderboard)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not leaderboard_path.exists():
        raise FileNotFoundError(f"Leaderboard not found: {leaderboard_path}")

    leaderboard_df = pd.read_csv(leaderboard_path).fillna("")
    benchmark_df = leaderboard_df[leaderboard_df["benchmark"] == args.benchmark]

    lines = []
    lines.append(f"# {args.benchmark.upper()} Experiment Report")
    lines.append("")
    lines.append("## Run Summary")
    lines.append("")

    summary_cols = [
        "timestamp",
        "run_name",
        "model_name",
        "limit",
        "accuracy",
        "prompt_path",
        "config_path",
    ]

    summary_cols = [c for c in summary_cols if c in benchmark_df.columns]

    lines.append(benchmark_df[summary_cols].to_markdown(index=False))
    lines.append("")
    lines.append("## Failure Type Summary")
    lines.append("")

    failure_rows = []

    for _, run in benchmark_df.iterrows():
        results_path = Path(run["output_dir"]) / "results.csv"

        if not results_path.exists():
            continue

        results_df = pd.read_csv(results_path).fillna("")

        if "failure_type" not in results_df.columns:
            continue

        counts = results_df["failure_type"].value_counts().to_dict()

        failure_rows.append(
            {
                "timestamp": run.get("timestamp", ""),
                "run_name": run.get("run_name", ""),
                "prompt_path": run.get("prompt_path", ""),
                "correct": counts.get("correct", 0),
                "no_numeric_answer": counts.get("no_numeric_answer", 0),
                "format_violation": counts.get("format_violation", 0),
                "wrong_numeric_answer": counts.get("wrong_numeric_answer", 0),
            }
        )

    if failure_rows:
        failure_df = pd.DataFrame(failure_rows)
        lines.append(failure_df.to_markdown(index=False))
    else:
        lines.append("No failure type data found.")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `tiny-gpt2` is only a smoke-test model.")
    lines.append("- SmolLM2-135M is instruction-tuned but still weak on GSM8K.")
    lines.append("- Prompt v1 currently has fewer formatting violations than prompt v2 strict.")

    output_path.write_text("\n".join(lines))

    print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
