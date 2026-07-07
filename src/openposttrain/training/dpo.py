from datasets import DatasetDict, load_dataset
from trl import DPOConfig


def load_dpo_dataset(train_path: str, validation_path: str) -> DatasetDict:
    # Already flat {"prompt", "chosen", "rejected"} strings (see
    # src/openposttrain/data/gsm8k_dpo.py) -- no conversion needed, unlike
    # the SFT dataset loader.
    return load_dataset(
        "json",
        data_files={"train": train_path, "validation": validation_path},
    )


def build_dpo_training_args(training_config: dict) -> DPOConfig:
    return DPOConfig(
        output_dir=training_config["output_dir"],
        num_train_epochs=training_config.get("num_train_epochs", 3),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 8),
        learning_rate=training_config.get("learning_rate", 1e-5),
        beta=training_config.get("beta", 0.1),
        max_length=training_config.get("max_length", 1024),
        logging_steps=training_config.get("logging_steps", 5),
        eval_strategy=training_config.get("eval_strategy", "epoch"),
        save_strategy=training_config.get("save_strategy", "epoch"),
        bf16=training_config.get("bf16", True),
        seed=training_config.get("seed", 42),
        report_to=training_config.get("report_to", "none"),
        # Same overfitting guardrail as SFT (Decision 019) -- keep whichever
        # epoch had the lowest eval_loss, not just the last one.
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
