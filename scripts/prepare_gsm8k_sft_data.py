import argparse
import json

from openposttrain.data.gsm8k_sft import prepare_gsm8k_sft_split
from openposttrain.utils.config import load_yaml_config
from openposttrain.utils.jsonl import write_jsonl
from openposttrain.utils.prompts import load_prompt_template


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a GSM8K SFT dataset in chat-message JSONL format."
    )
    parser.add_argument("--config", required=True, help="Path to data-prep YAML config.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml_config(args.config)

    dataset_config = config["dataset"]
    splits_config = config["splits"]
    limits_config = config["limits"]
    prompt_config = config["prompt"]
    output_config = config["output"]

    subset = dataset_config.get("subset", "main")
    prompt_template = load_prompt_template(prompt_config["path"])

    train_limit = limits_config.get("train_limit")
    validation_limit = limits_config.get("validation_limit")

    train_records = prepare_gsm8k_sft_split(
        subset=subset,
        split=splits_config["train_split"],
        prompt_template=prompt_template,
        limit=train_limit,
        offset=0,
    )
    write_jsonl(train_records, output_config["train_path"])

    val_records = prepare_gsm8k_sft_split(
        subset=subset,
        split=splits_config["validation_split"],
        prompt_template=prompt_template,
        limit=validation_limit,
        offset=train_limit or 0,
    )
    write_jsonl(val_records, output_config["validation_path"])

    print("SFT data preparation complete.")
    print(f"Train examples:      {len(train_records)} -> {output_config['train_path']}")
    print(f"Validation examples: {len(val_records)} -> {output_config['validation_path']}")
    print()
    print("Sample train record:")
    print(json.dumps(train_records[0], indent=2))


if __name__ == "__main__":
    main()
