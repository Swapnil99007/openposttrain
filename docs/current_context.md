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
Baseline eval, SFT data prep, LoRA SFT training, and base-vs-SFT evaluation are all done, including four rounds of diagnosis (overfitting fix, data scale-up + gentler LoRA, precision control, full-dataset scale-up). Final result: SFT adapter reaches 0.57 accuracy (7000 examples, bf16) vs. 0.70 baseline -- data quantity has hit diminishing returns (1500->7000 only gained +2pts). Open decision: try fixing training data quality/style, or move forward.

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

## Base vs SFT Result (final, across all experiments)

| Experiment | Train examples | Eval dtype | Accuracy |
|---|---:|---|---:|
| Baseline | - | fp16/bf16 | 0.70 |
| v1 | 200 | fp16 | 0.45 |
| v2 | 1500 | bf16 | 0.55 |
| v3 | 7000 | bf16 | 0.57 |

Diagnostic history: v1 (200 examples, r=16) regressed to 0.45 with clear training-time overfitting -> fixed overfitting via `load_best_model_at_end` (Decision 019) -> v2 (1500 examples, gentler LoRA r=8/alpha=16/lr=1e-4) still only reached 0.49, with flatter training curves but oddly incoherent failure examples -> found and controlled for an fp16-eval/bf16-train precision mismatch (base model is precision-robust at 0.70 either way; the adapter is not, 0.49->0.55) -> v3 (full ~7000-example train set, same settings) reached only 0.57, a diminishing-returns +2pts for 4.67x more data. Data quantity alone is unlikely to close the remaining gap. Full detail in `DECISIONS.md` Decision 020.

## Next Step
Open decision: try regenerating SFT targets from the base model's own verified-correct reasoning instead of GSM8K's terse gold text (untested remaining lever), vs. move forward to other pipeline stages and revisit later.

Targeted failure-mode SFT data (hand-curated, based on the Qwen failure patterns below) is still a separate, later addition on top of the general GSM8K data.
