import argparse
import json

from openposttrain.data.gsm8k_dpo import build_dpo_pairs
from openposttrain.models.hf_model import HFModel
from openposttrain.utils.config import load_yaml_config
from openposttrain.utils.jsonl import write_jsonl


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate GSM8K DPO preference pairs from an SFT'd model's own completions."
    )
    parser.add_argument("--config", required=True, help="Path to DPO data-prep YAML config.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml_config(args.config)

    model_config = config["model"]
    data_config = config["data"]

    model = HFModel(
        model_name=model_config["name"],
        device=model_config.get("device", "auto"),
        adapter_path=model_config.get("adapter_path"),
        dtype=model_config.get("dtype"),
    )

    train_limit = data_config["train_limit"]
    validation_limit = data_config["validation_limit"]
    max_new_tokens = model_config.get("max_new_tokens", 512)

    train_pairs = build_dpo_pairs(
        sft_jsonl_path=data_config["source_jsonl"],
        model=model,
        limit=train_limit,
        offset=0,
        max_new_tokens=max_new_tokens,
    )
    write_jsonl(train_pairs, data_config["train_output"])

    val_pairs = build_dpo_pairs(
        sft_jsonl_path=data_config["source_jsonl"],
        model=model,
        limit=validation_limit,
        offset=train_limit,
        max_new_tokens=max_new_tokens,
    )
    write_jsonl(val_pairs, data_config["validation_output"])

    print("DPO preference data generation complete.")
    print(f"Train pairs:      {len(train_pairs)} (from {train_limit} candidates) -> {data_config['train_output']}")
    print(
        f"Validation pairs: {len(val_pairs)} (from {validation_limit} candidates) -> "
        f"{data_config['validation_output']}"
    )
    print()
    print("Sample pair:")
    print(json.dumps(train_pairs[0] if train_pairs else {}, indent=2))


if __name__ == "__main__":
    main()
