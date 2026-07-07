import json
from pathlib import Path

from tqdm import tqdm

from openposttrain.evals.gsm8k import extract_model_answer
from openposttrain.models.hf_model import HFModel


def build_dpo_pairs(
    sft_jsonl_path: str,
    model: HFModel,
    limit: int,
    offset: int = 0,
    max_new_tokens: int = 512,
) -> list[dict]:
    """
    For each SFT training record in [offset, offset + limit), generate the
    current model's own completion for the same prompt. Where the model's
    own answer is wrong, build a DPO preference pair: chosen = the
    already-cleaned gold reasoning from the SFT data, rejected = the model's
    own (wrong) completion.

    On-policy: rejected completions come from the model's actual current
    behavior, not an arbitrary "bad" answer -- this targets DPO training
    directly at real, current failure modes.
    """
    records = []
    with Path(sft_jsonl_path).open() as f:
        for line in f:
            records.append(json.loads(line))

    records = records[offset : offset + limit]

    pairs = []

    for record in tqdm(records, desc=f"Generating DPO candidates [{offset}:{offset + limit}]"):
        prompt = record["messages"][0]["content"]
        chosen = record["messages"][1]["content"]
        gold_answer = record["metadata"]["final_answer"]

        model_completion = model.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )
        model_answer = extract_model_answer(model_completion)

        if model_answer != gold_answer:
            pairs.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": model_completion,
                }
            )

    return pairs
