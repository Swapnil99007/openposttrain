import re

from datasets import load_dataset
from tqdm import tqdm

from openposttrain.evals.gsm8k import extract_gsm8k_gold
from openposttrain.utils.prompts import format_prompt

CALCULATOR_ANNOTATION_PATTERN = re.compile(r"<<[^>]+>>")


def clean_gsm8k_reasoning(answer: str) -> str:
    """
    GSM8K solutions contain calculator annotations like '<<16-3=13>>13'
    and end with '#### <answer>'. Strip both so the reasoning reads like
    natural step-by-step text, with the final answer line added separately
    so it matches the eval's expected 'Final Answer: <number>' format.
    """
    reasoning = answer.split("####")[0]
    reasoning = CALCULATOR_ANNOTATION_PATTERN.sub("", reasoning)
    return reasoning.strip()


def build_sft_record(question: str, answer: str, prompt_template: str, split: str) -> dict:
    final_answer = extract_gsm8k_gold(answer)
    reasoning = clean_gsm8k_reasoning(answer)
    assistant_content = f"{reasoning}\n\nFinal Answer: {final_answer}"

    return {
        "messages": [
            {"role": "user", "content": format_prompt(prompt_template, question)},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "source": "gsm8k",
            "split": split,
            "final_answer": final_answer,
        },
    }


def prepare_gsm8k_sft_split(
    subset: str,
    split: str,
    prompt_template: str,
    limit: int | None = None,
    offset: int = 0,
) -> list[dict]:
    """
    Loads `split` of GSM8K and builds chat-format SFT records from rows
    [offset, offset + limit). The offset lets train/validation draw disjoint
    row ranges from the same split (e.g. both from "train") so they never
    overlap, without ever touching the "test" split used for benchmark eval.
    """
    dataset = load_dataset("openai/gsm8k", subset, split=split)

    end = offset + limit if limit else len(dataset)
    end = min(end, len(dataset))
    dataset = dataset.select(range(offset, end))

    records = []

    for row in tqdm(dataset, desc=f"Building GSM8K SFT records ({split}[{offset}:{end}])"):
        records.append(
            build_sft_record(
                question=row["question"],
                answer=row["answer"],
                prompt_template=prompt_template,
                split=split,
            )
        )

    return records
