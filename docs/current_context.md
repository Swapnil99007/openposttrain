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
Baseline eval, SFT data prep, LoRA SFT training, and base-vs-SFT evaluation are all done. The SFT adapter regressed GSM8K accuracy (70% -> 45%) despite fixing formatting -- diagnosing/fixing this (likely: training set too small) is the open next step.

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

## Base vs SFT Result

| | Baseline | SFT adapter |
|---|---:|---:|
| accuracy | 0.70 | 0.45 |
| format_violation | 18 | 3 |
| wrong_numeric_answer | 12 | 52 |

SFT fixed formatting compliance but substantially regressed reasoning accuracy. Likely cause: 200 training examples is too small/narrow to generalize from -- the adapter overfit formatting conventions while distorting general arithmetic reasoning (see Decision 020).

## Next Step
Open decision: scale up the SFT training set, reduce LoRA aggressiveness (lower rank/learning rate), and/or rule out an fp16-eval vs bf16-train dtype mismatch as a confound -- then re-run the same controlled comparison.

Targeted failure-mode SFT data (hand-curated, based on the Qwen failure patterns below) is still a separate, later addition on top of the general GSM8K data.
