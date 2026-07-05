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
Two full SFT tracks are now done and documented, with contrasting outcomes -- see below. Next candidates: Stage 19 (DPO) or synthetic/self-distilled data generation.

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

General-purpose GSM8K SFT data has been generated (not yet the targeted failure-mode data):

- `data/sft/gsm8k_train_small.jsonl` — 200 examples, GSM8K `train[0:200]`
- `data/sft/gsm8k_val_small.jsonl` — 50 examples, GSM8K `train[200:250]`

Both are chat-format JSONL (`{"messages": [...]}`), with the assistant turn ending in `Final Answer: <number>` to match the evaluator's parsing. GSM8K's `test` split was never touched, so it remains valid for an unbiased base-vs-SFT comparison later. See `docs/dataset_format.md`.

## LoRA SFT Training Complete

Trained a LoRA adapter (`outputs/sft/qwen2_5_1_5b_gsm8k_lora`) on `Qwen/Qwen2.5-1.5B-Instruct` using TRL's `SFTTrainer` + PEFT, 3 epochs over the 200/50 train/validation split.

| Epoch | eval_loss | eval_mean_token_accuracy |
|---:|---:|---:|
| 1 | 0.3808 | 0.8822 |
| 2 | 0.4005 | 0.8758 |
| 3 | 0.4545 | 0.8707 |

Overfitting after epoch 1 on this small dataset; `load_best_model_at_end` keeps the epoch-1 checkpoint automatically (verified via `md5sum`). `mean_token_accuracy` is a training proxy, not GSM8K exact-match accuracy.

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

## Next Step
Both tracks are documented -- together they show *when* SFT helps (teaching a genuinely missing skill) vs. hurts (overwriting existing tuned behavior with narrower data). Next candidates: Stage 19 (DPO) or synthetic/self-distilled data generation.

Targeted failure-mode SFT data (hand-curated, based on the Qwen failure patterns below) is still a separate, later addition on top of the general GSM8K data.
