import argparse

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from trl import DPOTrainer

from openposttrain.training.dpo import build_dpo_training_args, load_dpo_dataset
from openposttrain.utils.config import load_yaml_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DPO training, continuing an existing SFT LoRA adapter."
    )
    parser.add_argument("--config", required=True, help="Path to DPO training YAML config.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml_config(args.config)

    sft_adapter_path = config["model"]["sft_adapter_path"]
    data_config = config["data"]
    training_args = build_dpo_training_args(config["training"])

    # Load from the adapter directory, not the base model name, so we get
    # the exact tokenizer state saved alongside SFT training (same reasoning
    # as HFModel's adapter-path tokenizer loading fix).
    tokenizer = AutoTokenizer.from_pretrained(sft_adapter_path)
    tokenizer.padding_side = "left"  # required by DPOTrainer

    # Continue training the existing SFT adapter (is_trainable=True) rather
    # than starting a fresh peft_config -- TRL's documented pattern for
    # building on a prior PEFT run.
    model = AutoPeftModelForCausalLM.from_pretrained(
        sft_adapter_path,
        is_trainable=True,
        dtype=torch.bfloat16,
    )

    dataset = load_dpo_dataset(
        train_path=data_config["train_path"],
        validation_path=data_config["validation_path"],
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    train_result = trainer.train()
    trainer.save_model(training_args.output_dir)

    print("DPO training complete.")
    print(f"Adapter saved to: {training_args.output_dir}")
    print(f"Final train loss: {train_result.training_loss:.4f}")


if __name__ == "__main__":
    main()
