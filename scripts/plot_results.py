"""
Generates the summary charts embedded in README.md: accuracy progression,
failure-type breakdown, and LLM-judge win rate. Reads directly from the
existing results.csv / judge CSV files -- no new data collection, just
visualizing what's already documented in DECISIONS.md.
"""

import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = "docs/plots"

# (label, results.csv path, accuracy) -- accuracy is hardcoded from the
# documented, lineage-correct numbers (Decisions 021, 022, 027-028) rather
# than recomputed, since some of these runs' raw files were regenerated
# across different RunPod pods and the accuracy is the one already-verified
# number tying each stage to its place in the post-training arc.
STAGES = [
    ("Base\n(zero-shot)", "results/gsm8k/20260707_055109_gsm8k_qwen2_5_1_5b_base_sft_lora_512_tokens/results.csv", None),
    ("+ SFT", "results/gsm8k/20260707_055109_gsm8k_qwen2_5_1_5b_base_sft_lora_512_tokens/results.csv", 0.32),
    ("+ DPO", "results/gsm8k/20260707_044505_gsm8k_qwen2_5_1_5b_base_dpo_512_tokens/results.csv", 0.51),
    ("+ GRPO", "results/gsm8k/20260708_034912_gsm8k_qwen2_5_1_5b_base_grpo_512_tokens/results.csv", 0.52),
]

# Base zero-shot failure-type counts (Decision 022) -- no results.csv kept
# for this very first run, so hardcoded from the documented table.
BASE_COUNTS = {"correct": 3, "no_numeric_answer": 25, "format_violation": 0, "wrong_numeric_answer": 72}

FAILURE_TYPES = ["correct", "wrong_numeric_answer", "no_numeric_answer", "format_violation"]
COLORS = {
    "correct": "#2ca02c",
    "wrong_numeric_answer": "#d62728",
    "no_numeric_answer": "#7f7f7f",
    "format_violation": "#ff7f0e",
}


def plot_accuracy_progression():
    labels = ["Base\n(zero-shot)", "+ SFT", "+ DPO", "+ GRPO"]
    accuracies = [0.03, 0.32, 0.51, 0.52]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, accuracies, color=["#999999", "#1f77b4", "#1f77b4", "#1f77b4"])

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.015, f"{acc:.2f}", ha="center", fontweight="bold")

    ax.set_ylabel("GSM8K accuracy (100-example test split)")
    ax.set_ylim(0, 0.65)
    ax.set_title("Post-training arc: base model -> SFT -> DPO -> GRPO")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/accuracy_progression.png", dpi=150)
    plt.close(fig)


def plot_failure_types():
    labels = ["Base\n(zero-shot)", "+ SFT", "+ DPO", "+ GRPO"]

    counts_by_stage = [
        BASE_COUNTS,
        {k: v for k, v in pd.read_csv(STAGES[1][1]).fillna("")["failure_type"].value_counts().to_dict().items()},
        {k: v for k, v in pd.read_csv(STAGES[2][1]).fillna("")["failure_type"].value_counts().to_dict().items()},
        {k: v for k, v in pd.read_csv(STAGES[3][1]).fillna("")["failure_type"].value_counts().to_dict().items()},
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = [0] * len(labels)

    for failure_type in FAILURE_TYPES:
        values = [counts.get(failure_type, 0) for counts in counts_by_stage]
        ax.bar(labels, values, bottom=bottom, label=failure_type, color=COLORS[failure_type])
        bottom = [b + v for b, v in zip(bottom, values)]

    ax.set_ylabel("Count (out of 100 examples)")
    ax.set_title("Failure-type breakdown across the post-training arc")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/failure_types.png", dpi=150)
    plt.close(fig)


def plot_judge_win_rate():
    df = pd.read_csv("reports/judge_sft_vs_dpo.csv").fillna("")
    counts = df["winner"].value_counts()

    labels = ["SFT+DPO", "tie", "SFT"]
    values = [counts.get(label, 0) for label in labels]
    total = sum(values)
    colors = ["#1f77b4", "#999999", "#d62728"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(labels, values, color=colors)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3, f"{v}/{total} ({v / total:.0%})", ha="center", fontweight="bold")

    ax.set_ylabel("Number of questions")
    ax.set_title("LLM-as-judge: SFT vs. SFT+DPO reasoning quality (30 pairs)")
    ax.set_ylim(0, max(values) + 3)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/judge_win_rate.png", dpi=150)
    plt.close(fig)


def main():
    import os

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plot_accuracy_progression()
    plot_failure_types()
    plot_judge_win_rate()

    print(f"Saved 3 charts to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
