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
The project has completed: baseline evaluation, GSM8K SFT data preparation, and LoRA SFT training (verified working, adapter saved). Next is base-vs-SFT evaluation.

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

## Next Step
Stage 17: add PEFT adapter-loading support to the eval pipeline, then run the SFT adapter against the same GSM8K `test[0:100]` slice used for the Qwen baseline for a real base-vs-SFT accuracy comparison.

Targeted failure-mode SFT data (hand-curated, based on the Qwen failure patterns below) is still a separate, later addition on top of the general GSM8K data.
