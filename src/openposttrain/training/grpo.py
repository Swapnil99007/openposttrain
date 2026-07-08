from datasets import load_dataset
from trl import GRPOConfig


def load_grpo_dataset(train_path: str, validation_path: str):
    # Flat {"prompt", "ground_truth"} JSONL (see
    # src/openposttrain/data/gsm8k_grpo.py) -- no conversion needed.
    return load_dataset(
        "json",
        data_files={"train": train_path, "validation": validation_path},
    )


def build_grpo_training_args(training_config: dict) -> GRPOConfig:
    return GRPOConfig(
        output_dir=training_config["output_dir"],
        num_train_epochs=training_config.get("num_train_epochs", 1),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
        learning_rate=training_config.get("learning_rate", 1e-6),
        # Nonzero beta (unlike TRL's default 0.0) -- we're refining an
        # already-tuned DPO policy, not training from scratch, so keep a
        # small KL pull back toward it rather than letting GRPO drift freely.
        beta=training_config.get("beta", 0.02),
        num_generations=training_config.get("num_generations", 4),
        max_completion_length=training_config.get("max_completion_length", 400),
        temperature=training_config.get("temperature", 1.0),
        reward_weights=training_config.get("reward_weights", [0.8, 0.2]),
        remove_unused_columns=False,
        logging_steps=training_config.get("logging_steps", 1),
        eval_strategy=training_config.get("eval_strategy", "epoch"),
        save_strategy=training_config.get("save_strategy", "epoch"),
        bf16=training_config.get("bf16", True),
        seed=training_config.get("seed", 42),
        report_to=training_config.get("report_to", "none"),
        use_vllm=training_config.get("use_vllm", False),
    )
