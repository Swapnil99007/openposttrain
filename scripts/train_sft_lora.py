import argparse

from trl import SFTTrainer

from openposttrain.training.sft_lora import (
    build_lora_config,
    build_sft_training_args,
    load_sft_dataset,
)
from openposttrain.utils.config import load_yaml_config


def parse_args():
    parser = argparse.ArgumentParser(description="Run LoRA SFT training from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to training YAML config.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml_config(args.config)

    model_name = config["model"]["name"]
    data_config = config["data"]
    lora_config = build_lora_config(config.get("lora", {}))
    training_args = build_sft_training_args(config["training"])

    dataset = load_sft_dataset(
        train_path=data_config["train_path"],
        validation_path=data_config["validation_path"],
        conversational=data_config.get("conversational", True),
    )

    trainer = SFTTrainer(
        model=model_name,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora_config,
    )

    train_result = trainer.train()
    trainer.save_model(training_args.output_dir)

    print("LoRA SFT training complete.")
    print(f"Adapter saved to: {training_args.output_dir}")
    print(f"Final train loss: {train_result.training_loss:.4f}")


if __name__ == "__main__":
    main()
