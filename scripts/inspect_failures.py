import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect failed evaluation examples.")
    parser.add_argument(
        "--results",
        required=True,
        help="Path to a results.csv file from an evaluation run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of failed examples to print.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.results).fillna("")

    failed_df = df[df["is_correct"] == False]

    total = len(df)
    failed = len(failed_df)
    passed = total - failed

    print("=" * 80)
    print("Evaluation Failure Summary")
    print("=" * 80)
    print(f"Total examples: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Accuracy: {passed / total if total else 0:.4f}")

    if "failure_type" in df.columns:
        print()
        print("Failure types:")
        print(df["failure_type"].value_counts().to_string())

    print("=" * 80)

    for idx, row in failed_df.head(args.limit).iterrows():
        print()
        print("-" * 80)
        print(f"Example index: {idx}")
        print("-" * 80)

        if "failure_type" in row:
            print("\nFailure type:")
            print(row["failure_type"])

        print("\nQuestion:")
        print(row["question"])

        print("\nGold extracted answer:")
        print(row["extracted_gold"])

        print("\nModel extracted answer:")
        print(row["extracted_model"])

        print("\nModel full answer:")
        print(row["model_answer"])


if __name__ == "__main__":
    main()
