import argparse

import anthropic
import pandas as pd

from openposttrain.judge.llm_judge import judge_pair


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pairwise LLM-as-judge comparison between two eval runs' results.csv files."
    )
    parser.add_argument("--run-a", required=True, help="Path to results.csv for run A.")
    parser.add_argument("--run-b", required=True, help="Path to results.csv for run B.")
    parser.add_argument("--label-a", default="A", help="Display label for run A.")
    parser.add_argument("--label-b", default="B", help="Display label for run B.")
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Max number of questions to judge (cost control; default 30).",
    )
    parser.add_argument("--model", default="claude-opus-4-8", help="Judge model.")
    parser.add_argument("--output", required=True, help="Path to write per-question judge verdicts CSV.")
    return parser.parse_args()


def main():
    args = parse_args()

    df_a = pd.read_csv(args.run_a).fillna("")
    df_b = pd.read_csv(args.run_b).fillna("")

    if len(df_a) != len(df_b):
        raise ValueError(f"Run A has {len(df_a)} rows, run B has {len(df_b)} -- expected matching runs.")

    client = anthropic.Anthropic()

    rows = []
    win_counts = {args.label_a: 0, args.label_b: 0, "tie": 0}

    limit = min(args.limit, len(df_a))

    for idx in range(limit):
        row_a = df_a.iloc[idx]
        row_b = df_b.iloc[idx]

        if row_a["question"] != row_b["question"]:
            raise ValueError(
                f"Row {idx}: question mismatch between run A and run B -- are these the same eval set?"
            )

        verdict = judge_pair(
            client=client,
            question=row_a["question"],
            gold_answer=row_a["gold_answer"],
            answer_a=row_a["model_answer"],
            answer_b=row_b["model_answer"],
            model=args.model,
        )

        winner_label = {"A": args.label_a, "B": args.label_b, "tie": "tie"}[verdict.winner]
        win_counts[winner_label] += 1

        rows.append(
            {
                "question": row_a["question"],
                "winner": winner_label,
                "reasoning_quality_score": verdict.reasoning_quality_score,
                "explanation": verdict.explanation,
            }
        )

        print(f"[{idx + 1}/{limit}] winner={winner_label} score={verdict.reasoning_quality_score}")

    pd.DataFrame(rows).to_csv(args.output, index=False)

    print()
    print("=" * 60)
    print("Judge Summary")
    print("=" * 60)
    for label, count in win_counts.items():
        print(f"{label}: {count}/{limit} ({count / limit:.1%})")
    print(f"Saved per-question verdicts to: {args.output}")


if __name__ == "__main__":
    main()
