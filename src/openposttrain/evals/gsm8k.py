import re
from dataclasses import dataclass
from datasets import load_dataset
from tqdm import tqdm


@dataclass
class EvalResult:
    question: str
    gold_answer: str
    model_answer: str
    extracted_gold: str
    extracted_model: str
    is_correct: bool
    failure_type: str


def extract_gsm8k_gold(answer: str) -> str:
    """
    GSM8K answers usually contain the final answer after ####.
    Example: '... #### 42'
    """
    if "####" in answer:
        return answer.split("####")[-1].strip().replace(",", "")

    numbers = re.findall(r"-?\d+\.?\d*", answer.replace(",", ""))
    return numbers[-1] if numbers else ""


def extract_model_answer(response: str) -> str:
    """
    Extract the number after 'Final Answer:' if present.
    Otherwise extract the last number from the response.
    """
    cleaned_response = response.replace(",", "")

    final_answer_match = re.search(
        r"Final Answer:\s*(-?\d+\.?\d*)",
        cleaned_response,
        re.IGNORECASE,
    )

    if final_answer_match:
        return final_answer_match.group(1).strip()

    numbers = re.findall(r"-?\d+\.?\d*", cleaned_response)
    return numbers[-1] if numbers else ""


def classify_result(
    extracted_gold: str,
    extracted_model: str,
    model_answer: str,
) -> str:
    if extracted_gold == extracted_model:
        return "correct"

    if not extracted_model:
        return "no_numeric_answer"

    if "final answer" not in model_answer.lower():
        return "format_violation"

    return "wrong_numeric_answer"


def build_prompt(question: str) -> str:
    return f"""
Solve the following math problem step by step.

At the end, write the final answer in this exact format:

Final Answer: <number>

Problem:
{question}
""".strip()


def run_gsm8k_eval(
    model,
    split: str = "test",
    limit: int = 50,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
):
    dataset = load_dataset("openai/gsm8k", "main", split=split)

    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    results = []

    for row in tqdm(dataset, desc="Evaluating GSM8K"):
        question = row["question"]
        gold_answer = row["answer"]

        prompt = build_prompt(question)
        model_answer = model.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

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
                question=question,
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
    }
