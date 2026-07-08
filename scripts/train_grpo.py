import argparse

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from trl import GRPOTrainer

from openposttrain.training.grpo import build_grpo_training_args, load_grpo_dataset
from openposttrain.training.grpo_rewards import accuracy_reward, format_reward
from openposttrain.utils.config import load_yaml_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GRPO training, continuing an existing DPO LoRA adapter."
    )
    parser.add_argument("--config", required=True, help="Path to GRPO training YAML config.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml_config(args.config)

    dpo_adapter_path = config["model"]["dpo_adapter_path"]
    data_config = config["data"]
    training_args = build_grpo_training_args(config["training"])

    # Load from the adapter directory, not the base model name, so we get
    # the exact tokenizer state saved alongside DPO training (same reasoning
    # as HFModel's and train_dpo.py's adapter-path tokenizer loading).
    tokenizer = AutoTokenizer.from_pretrained(dpo_adapter_path)
    tokenizer.padding_side = "left"

    # Continue training the existing DPO adapter (is_trainable=True) rather
    # than starting a fresh peft_config -- same pattern as DPO continuing SFT.
    model = AutoPeftModelForCausalLM.from_pretrained(
        dpo_adapter_path,
        is_trainable=True,
        dtype=torch.bfloat16,
    )

    dataset = load_grpo_dataset(
        train_path=data_config["train_path"],
        validation_path=data_config["validation_path"],
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=[accuracy_reward, format_reward],
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    train_result = trainer.train()
    trainer.save_model(training_args.output_dir)

    print("GRPO training complete.")
    print(f"Adapter saved to: {training_args.output_dir}")
    print(f"Final train loss: {train_result.training_loss:.4f}")


if __name__ == "__main__":
    main()
