from datasets import load_dataset
from tqdm import tqdm

from openposttrain.evals.gsm8k import extract_gsm8k_gold
from openposttrain.utils.prompts import format_prompt


def prepare_gsm8k_grpo_split(
    subset: str,
    split: str,
    prompt_template: str,
    limit: int,
    offset: int = 0,
) -> list[dict]:
    """
    GRPO grades the model's own live generation against ground_truth, so
    unlike SFT/DPO data prep, no gold reasoning text is needed here -- just
    (prompt, ground_truth) pairs.
    """
    dataset = load_dataset("openai/gsm8k", subset, split=split)

    end = min(offset + limit, len(dataset))
    dataset = dataset.select(range(offset, end))

    records = []

    for row in tqdm(dataset, desc=f"Building GSM8K GRPO records ({split}[{offset}:{end}])"):
        records.append(
            {
                "prompt": format_prompt(prompt_template, row["question"]),
                "ground_truth": extract_gsm8k_gold(row["answer"]),
            }
        )

    return records
