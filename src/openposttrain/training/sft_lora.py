from datasets import DatasetDict, load_dataset
from peft import LoraConfig
from trl import SFTConfig


def to_prompt_completion(example: dict) -> dict:
    """
    Our data files store {"messages": [user, assistant]} (see
    src/openposttrain/data/gsm8k_sft.py). TRL's SFTTrainer masks loss to
    the completion automatically for "prompt-completion" format datasets,
    with no chat-template requirements beyond what the tokenizer already
    has -- unlike assistant_only_loss, which needs {% generation %}
    markers in the chat template that aren't guaranteed to exist for
    every model family. Splitting into prompt/completion here avoids
    relying on that.
    """
    messages = example["messages"]
    return {
        "prompt": [messages[0]],
        "completion": [messages[1]],
    }


def load_sft_dataset(train_path: str, validation_path: str) -> DatasetDict:
    dataset = load_dataset(
        "json",
        data_files={"train": train_path, "validation": validation_path},
    )
    return dataset.map(to_prompt_completion, remove_columns=dataset["train"].column_names)


def build_lora_config(lora_config: dict) -> LoraConfig:
    return LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("alpha", 32),
        lora_dropout=lora_config.get("dropout", 0.05),
        target_modules=lora_config.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        task_type="CAUSAL_LM",
    )


def build_sft_training_args(training_config: dict) -> SFTConfig:
    return SFTConfig(
        output_dir=training_config["output_dir"],
        num_train_epochs=training_config.get("num_train_epochs", 3),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 8),
        learning_rate=training_config.get("learning_rate", 2e-4),
        max_length=training_config.get("max_length", 768),
        logging_steps=training_config.get("logging_steps", 5),
        eval_strategy=training_config.get("eval_strategy", "epoch"),
        save_strategy=training_config.get("save_strategy", "epoch"),
        bf16=training_config.get("bf16", True),
        seed=training_config.get("seed", 42),
        report_to=training_config.get("report_to", "none"),
    )
