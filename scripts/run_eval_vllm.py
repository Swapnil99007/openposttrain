import argparse
import json
import os
from datetime import datetime

import pandas as pd
from vllm import LLM
from vllm.lora.request import LoRARequest

from openposttrain.serving.vllm_eval import run_gsm8k_eval_vllm
from openposttrain.utils.config import load_yaml_config
from openposttrain.utils.leaderboard import append_to_leaderboard
from openposttrain.utils.prompts import load_prompt_template


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GSM8K evaluation via vLLM (batched serving comparison)."
    )
    parser.add_argument("--config", required=True, help="Path to evaluation YAML config.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml_config(args.config)

    model_config = config["model"]
    benchmark_config = config["benchmark"]
    output_config = config["output"]

    model_name = model_config["name"]
    adapter_path = model_config.get("adapter_path")
    prompt_path = benchmark_config["prompt_path"]

    prompt_template = load_prompt_template(prompt_path)

    llm = LLM(
        model=model_name,
        enable_lora=adapter_path is not None,
        max_lora_rank=model_config.get("max_lora_rank", 8),
        dtype=model_config.get("dtype", "bfloat16"),
    )

    lora_request = LoRARequest("adapter", 1, adapter_path) if adapter_path else None

    eval_output = run_gsm8k_eval_vllm(
        llm=llm,
        prompt_template=prompt_template,
        split=benchmark_config.get("split", "test"),
        limit=benchmark_config.get("limit"),
        max_new_tokens=model_config.get("max_new_tokens", 512),
        temperature=model_config.get("temperature", 0.0),
        top_p=model_config.get("top_p", 1.0),
        lora_request=lora_request,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config.get("run_name", f"gsm8k_vllm_{timestamp}")

    output_dir = os.path.join(
        output_config.get("base_dir", "results"),
        "gsm8k",
        f"{timestamp}_{run_name}",
    )
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "timestamp": timestamp,
        "run_name": run_name,
        "model_name": model_name,
        "benchmark": eval_output["benchmark"],
        "split": eval_output["split"],
        "limit": eval_output["limit"],
        "accuracy": eval_output["accuracy"],
        "num_examples": eval_output["num_examples"],
        "max_new_tokens": model_config.get("max_new_tokens", 512),
        "temperature": model_config.get("temperature", 0.0),
        "top_p": model_config.get("top_p", 1.0),
        "prompt_path": prompt_path,
        "adapter_path": adapter_path,
        "dtype": model_config.get("dtype"),
        "config_path": args.config,
        "output_dir": output_dir,
        "wall_clock_seconds": eval_output["wall_clock_seconds"],
        "total_output_tokens": eval_output["total_output_tokens"],
        "tokens_per_second": eval_output["tokens_per_second"],
        "serving_backend": "vllm",
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
                "failure_type": r.failure_type,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(f"{output_dir}/results.csv", index=False)

    leaderboard_path = output_config.get(
        "leaderboard_path",
        os.path.join(output_config.get("base_dir", "results"), "leaderboard.csv"),
    )

    append_to_leaderboard(
        leaderboard_path=leaderboard_path,
        row=summary,
    )

    print("vLLM evaluation complete.")
    print(json.dumps(summary, indent=2))
    print(f"Saved results to: {output_dir}")
    print(f"Updated leaderboard: {leaderboard_path}")


if __name__ == "__main__":
    main()
