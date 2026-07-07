# Current Project Context

## Project
OpenPostTrain is an end-to-end LLM post-training project.

The goal is to build practical experience with:

- evaluation
- failure analysis
- SFT data preparation
- LoRA supervised fine-tuning
- base vs fine-tuned model comparison
- later preference tuning / DPO

## Current Stage
The full post-training arc is done: baseline -> SFT -> DPO, with the base-model track as the headline result (0.03 -> 0.37 -> 0.51). See below for the complete picture. Next candidates: serving/inference comparison, or synthetic/self-distilled data generation.

## Current Baseline

Model:

- `Qwen/Qwen2.5-1.5B-Instruct`

Benchmark:

- GSM8K
- test split
- first 100 examples

Hardware:

- RunPod RTX 3090

Prompt:

- `prompts/gsm8k_v1.txt`

Generation:

- deterministic
- `temperature=0.0`
- `top_p=1.0`
- `max_new_tokens=512`

## Baseline Results

| Run | Max New Tokens | Accuracy | Correct | Format Violations | Wrong Numeric |
|---|---:|---:|---:|---:|---:|
| Qwen baseline | 256 | 0.43 | 43 | 54 | 3 |
| Qwen baseline | 512 | 0.70 | 70 | 18 | 12 |

## Main Finding
The 256-token Qwen run was not a valid final baseline because many generations were truncated.

The 512-token Qwen run is the current meaningful baseline.

## Current Failure Pattern
Remaining failures include:

- arithmetic errors
- word-problem interpretation errors
- boundary condition errors
- final-answer formatting issues

## SFT Data Prepared

GSM8K SFT data generated at three sizes as the post-training investigation scaled up: `data/sft/gsm8k_train_small.jsonl` (200, used for Instruct-track v1), `..._medium.jsonl` (1500, v2), `..._full.jsonl` (7000, v3 and the base-model track). All are chat-format JSONL (`{"messages": [...]}`), assistant turn ending in `Final Answer: <number>` to match the evaluator's parsing. GSM8K's `test` split was never touched by any of these, so it stayed valid for base-vs-SFT comparison throughout. See `docs/dataset_format.md`.

## SFT Track 1: Instruct Model (Decision 020) -- documented regression

Fine-tuning `Qwen/Qwen2.5-1.5B-Instruct` (already heavily instruction-tuned) only ever regressed accuracy, across four rounds of fixes:

| Experiment | Train examples | Eval dtype | Accuracy |
|---|---:|---|---:|
| Baseline (no adapter) | - | fp16/bf16 | 0.70 |
| v1 | 200 | fp16 | 0.45 |
| v2 | 1500 | bf16 | 0.55 |
| v3 | 7000 | bf16 | 0.57 |

Root cause: fine-tuning an already-tuned model with narrower/lower-quality data tends to overwrite good behavior rather than add new capability. Data quantity alone showed clear diminishing returns (1500->7000 only gained +2pts). Full diagnostic history in `DECISIONS.md` Decision 020.

## SFT Track 2: Base Model (Decision 021) -- headline success

Pivoted to fine-tuning `Qwen/Qwen2.5-1.5B` (base, non-instruct) instead -- a model with no existing GSM8K ability to overwrite:

| | Raw base model (zero-shot) | Base + SFT |
|---|---:|---:|
| accuracy | 0.03 (functionally ~0, almost entirely degenerate repetition) | **0.37** |

A real, dramatic, qualitative improvement: the raw model doesn't attempt the task at all (echoes the prompt, then repeats a junk token); the SFT'd model reliably formats answers and mostly reasons correctly, with most "wrong" answers being genuine single-step arithmetic slips rather than gibberish.

Two real bugs found and fixed along the way: (1) borrowing the Instruct model's chat template crashed on an unresolved PEFT/tied-embeddings interaction -- fixed by training on plain text instead of chat-formatted messages; (2) the same `repetition_penalty` settings used to diagnose the raw model's degenerate decoding were, when kept "for a controlled comparison," actively sabotaging the fine-tuned model's legitimate generation -- removing them for the SFT'd model's eval was the correct call, verified by inspecting completions. Full diagnostic path in `DECISIONS.md` Decision 021.

## Stage 19: DPO on the SFT'd Base Model (Decision 022) -- further improvement

Continued the SFT LoRA adapter with DPO, using on-policy preference pairs: for training questions the SFT'd model gets wrong, `chosen` = gold reasoning, `rejected` = the model's own actual wrong completion (targets real, current failure modes rather than generic bad answers).

| Run | Correct | no_numeric_answer | format_violation | wrong_numeric_answer | Accuracy |
|---|---:|---:|---:|---:|---:|
| Base zero-shot | 3 | 25 | 0 | 72 | 0.03 |
| Base + SFT | 37 | 10 | 1 | 52 | 0.37 |
| **Base + SFT + DPO** | **51** | 10 | 3 | 36 | **0.51** |

A real, controlled +14-point improvement, concentrated specifically in fixing close-but-wrong reasoning (`wrong_numeric_answer`: 52 -> 36); the degenerate-generation failure mode (`no_numeric_answer`) was untouched. Training note: `eval_loss` decreased monotonically across all 3 epochs (no overfitting, unlike every SFT run). Both the SFT and DPO adapters were transferred off the ephemeral RunPod pod to the Mac (`outputs/sft/`, `outputs/dpo/`, both gitignored) so future pods can re-upload rather than retrain.

## Next Step
The full post-training arc (baseline 0.03 -> SFT 0.37 -> DPO 0.51) is complete and documented end to end -- a strong, honest interview story showing both when SFT hurts (Instruct track) and when SFT + DPO together genuinely help (base-model track). Next candidates: serving/inference comparison (vLLM/TensorRT-LLM, ties to existing background), or synthetic/self-distilled data generation to push further.

Targeted failure-mode SFT data (hand-curated, based on the Qwen failure patterns below) is still a separate, later addition on top of the general GSM8K data.
