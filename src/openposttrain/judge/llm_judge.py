import anthropic

from openposttrain.judge.schemas import JudgeVerdict

JUDGE_SYSTEM_PROMPT = """You are grading two candidate solutions to the same grade-school math word problem.

Given the question, the correct (gold) solution, and two candidate answers (A and B), decide which
candidate's reasoning is more correct and better explained. A candidate can reach the right final
number through flawed reasoning (score it down) or the wrong number despite mostly sound reasoning
(note the specific error in your explanation). If both are equally good or equally bad, say "tie".

reasoning_quality_score is 1-10, for the winning candidate's reasoning quality (10 = flawless,
well-explained reasoning; 1 = incoherent or nonsensical). For a "tie", score reflects both equally.
"""


def build_judge_prompt(question: str, gold_answer: str, answer_a: str, answer_b: str) -> str:
    return (
        f"Question:\n{question}\n\n"
        f"Gold solution:\n{gold_answer}\n\n"
        f"Candidate A:\n{answer_a}\n\n"
        f"Candidate B:\n{answer_b}\n\n"
        "Which candidate's reasoning is better, and why?"
    )


def judge_pair(
    client: anthropic.Anthropic,
    question: str,
    gold_answer: str,
    answer_a: str,
    answer_b: str,
    model: str = "claude-opus-4-8",
    max_retries: int = 2,
) -> JudgeVerdict:
    """
    Runs one pairwise comparison. Retries on malformed/failed responses
    rather than silently dropping them -- a transient parse failure
    shouldn't quietly disappear from the aggregate win-rate.
    """
    prompt = build_judge_prompt(question, gold_answer, answer_a, answer_b)

    last_error = None
    for _ in range(max_retries + 1):
        try:
            response = client.messages.parse(
                model=model,
                max_tokens=1024,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                output_format=JudgeVerdict,
            )
            return response.parsed_output
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Judge failed after {max_retries + 1} attempts: {last_error}")
