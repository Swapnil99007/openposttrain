import json
import os
from datetime import datetime

import pandas as pd

from openposttrain.models.hf_model import HFModel
from openposttrain.evals.gsm8k import run_gsm8k_eval


def main():
    model_name = "sshleifer/tiny-gpt2"

    model = HFModel(model_name=model_name)

    eval_output = run_gsm8k_eval(
        model=model,
        split="test",
        limit=3,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/gsm8k/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "model_name": model_name,
        "benchmark": eval_output["benchmark"],
        "split": eval_output["split"],
        "limit": eval_output["limit"],
        "accuracy": eval_output["accuracy"],
        "num_examples": eval_output["num_examples"],
    }

    with open(f"{output_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    rows = []

    for r in eval_output["results"]:
        rows.append(
            {
                "question": r.question,
                "gold_answer": r.gold_answer,
                "model_answer": r.model_answer,
                "extracted_gold": r.extracted_gold,
                "extracted_model": r.extracted_model,
                "is_correct": r.is_correct,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(f"{output_dir}/results.csv", index=False)

    print("Evaluation complete.")
    print(json.dumps(summary, indent=2))
    print(f"Saved results to: {output_dir}")


if __name__ == "__main__":
    main()
