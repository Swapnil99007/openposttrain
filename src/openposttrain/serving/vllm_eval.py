import time

from datasets import load_dataset

from openposttrain.evals.gsm8k import EvalResult, classify_result, extract_gsm8k_gold, extract_model_answer
from openposttrain.utils.prompts import format_prompt


def run_gsm8k_eval_vllm(
    llm,
    prompt_template: str,
    split: str = "test",
    limit: int = 100,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    lora_request=None,
):
    """
    Same GSM8K task and scoring logic as run_gsm8k_eval (evals/gsm8k.py),
    but generates all prompts in a single batched vLLM call instead of one
    at a time -- this is the actual throughput comparison against the
    sequential HFModel.generate() loop.
    """
    from vllm import SamplingParams

    dataset = load_dataset("openai/gsm8k", "main", split=split)

    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    prompts = [format_prompt(prompt_template, row["question"]) for row in dataset]

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    wall_clock_seconds = time.perf_counter() - start

    results = []
    total_output_tokens = 0

    for row, output in zip(dataset, outputs):
        model_answer = output.outputs[0].text.strip()
        total_output_tokens += len(output.outputs[0].token_ids)

        gold_answer = row["answer"]
        extracted_gold = extract_gsm8k_gold(gold_answer)
        extracted_model = extract_model_answer(model_answer)

        is_correct = extracted_gold == extracted_model
        failure_type = classify_result(
            extracted_gold=extracted_gold,
            extracted_model=extracted_model,
            model_answer=model_answer,
        )

        results.append(
            EvalResult(
                question=row["question"],
                gold_answer=gold_answer,
                model_answer=model_answer,
                extracted_gold=extracted_gold,
                extracted_model=extracted_model,
                is_correct=is_correct,
                failure_type=failure_type,
            )
        )

    accuracy = sum(r.is_correct for r in results) / len(results)

    return {
        "benchmark": "gsm8k",
        "split": split,
        "limit": limit,
        "accuracy": accuracy,
        "num_examples": len(results),
        "results": results,
        "wall_clock_seconds": wall_clock_seconds,
        "total_output_tokens": total_output_tokens,
        "tokens_per_second": total_output_tokens / wall_clock_seconds if wall_clock_seconds > 0 else 0.0,
    }
