from openposttrain.evals.gsm8k import extract_model_answer

# Reuses the exact grading logic already trusted from the GSM8K evaluator
# (Decision 021) and the LLM-judge comparison input (Decision 025), rather
# than writing new grading code for the RL reward signal.


def accuracy_reward(completions: list[str], ground_truth: list[str], **kwargs) -> list[float]:
    return [
        1.0 if extract_model_answer(completion) == gt else 0.0
        for completion, gt in zip(completions, ground_truth)
    ]


def format_reward(completions: list[str], **kwargs) -> list[float]:
    return [1.0 if "final answer" in completion.lower() else 0.0 for completion in completions]
