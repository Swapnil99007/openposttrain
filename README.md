# OpenPostTrain

OpenPostTrain is an end-to-end LLM post-training platform.

The goal is to learn and demonstrate practical LLM post-training workflows, including:

- LLM evaluation
- LLM-as-a-judge
- Synthetic data generation
- Data filtering and curation
- Supervised fine-tuning
- Preference optimization
- Model comparison
- Inference deployment

## Current Status

The project currently supports a config-driven GSM8K evaluation pipeline.

A tiny model is used locally for smoke testing so the full pipeline can be tested without downloading large LLMs.

## Setup

Create and activate a virtual environment:

    python3 -m venv .venv
    source .venv/bin/activate

Install dependencies:

    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

## Run GSM8K Smoke Test

    PYTHONPATH=src python scripts/run_eval.py --config configs/eval_gsm8k_tiny.yaml

Expected behavior:

- Loads sshleifer/tiny-gpt2
- Loads GSM8K test split
- Evaluates 3 examples
- Saves summary.json
- Saves results.csv

## Current Architecture

YAML Config
    |
    v
Evaluation Runner
    |
    v
HuggingFace Model Wrapper
    |
    v
GSM8K Evaluator
    |
    v
summary.json + results.csv

## Notes

sshleifer/tiny-gpt2 is only used for smoke testing. It is not expected to achieve meaningful accuracy.

Real evaluations will use stronger models such as Qwen or Llama on RunPod or external model storage.

## Leaderboard

Each evaluation run appends a summary row to:

    results/leaderboard.csv

This file tracks:

- model name
- benchmark
- split
- number of examples
- accuracy
- config path
- output directory

The leaderboard is local-only for now because `results/` is ignored by Git.

## Inspect Failed Examples

After running an evaluation, inspect failed examples with:

    python scripts/inspect_failures.py --results path/to/results.csv --limit 5

Example:

    python scripts/inspect_failures.py --results results/gsm8k/<run_dir>/results.csv --limit 3

This helps identify whether failures come from model reasoning, prompt formatting, or answer extraction.

## GSM8K Failure Types

The GSM8K evaluator records a `failure_type` for each example:

- `correct`
- `no_numeric_answer`
- `format_violation`
- `wrong_numeric_answer`

These categories make it easier to understand why a model failed.

## Generation Settings

Evaluation configs include generation parameters:

    max_new_tokens: 256
    temperature: 0.0
    top_p: 1.0

These control how the model generates answers.

For benchmark evaluation, deterministic generation is preferred, so temperature is usually set to 0.0.

## Inspect Latest Failed Examples

To inspect the latest run for a benchmark:

    python scripts/inspect_latest_failures.py --benchmark gsm8k --limit 3

## Run Small Instruction Model Locally

To run GSM8K with a small instruction-tuned model:

    PYTHONPATH=src python scripts/run_eval.py --config configs/eval_gsm8k_smollm2_135m.yaml

This uses:

    HuggingFaceTB/SmolLM2-135M-Instruct

This model is still small and not expected to be strong on GSM8K, but it is more useful than tiny-gpt2 for testing instruction following.

## Prompt Templates

Prompt templates are stored under:

    prompts/

Current GSM8K prompt templates:

    prompts/gsm8k_v1.txt
    prompts/gsm8k_v2_strict.txt

Evaluation configs select a prompt with:

    prompt_path: prompts/gsm8k_v2_strict.txt

## Compare Runs

To compare evaluation runs from the leaderboard:

    python scripts/compare_runs.py --benchmark gsm8k

## Compare Failure Types

To compare failure categories across runs:

    python scripts/compare_failure_types.py --benchmark gsm8k

## Clean Prompt Comparison

To run SmolLM2 with GSM8K prompt v1:

    PYTHONPATH=src python scripts/run_eval.py --config configs/eval_gsm8k_smollm2_135m_v1.yaml

Then compare failure types:

    python scripts/compare_failure_types.py --benchmark gsm8k

## Compare Prompt Versions

To compare prompt versions for a model:

    python scripts/compare_prompts.py --benchmark gsm8k --model-name HuggingFaceTB/SmolLM2-135M-Instruct

## Generate Experiment Report

To generate a markdown report for GSM8K:

    python scripts/generate_experiment_report.py --benchmark gsm8k --output reports/gsm8k_report.md

## Current GSM8K Baseline

The current meaningful baseline uses:

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Benchmark: GSM8K
- Split: test
- Limit: 100
- Hardware: RunPod RTX 3090
- Prompt: `prompts/gsm8k_v1.txt`
- Generation: deterministic, `temperature=0.0`

### Results

| Model | Max New Tokens | Accuracy |
|---|---:|---:|
| Qwen2.5-1.5B-Instruct | 256 | 0.43 |
| Qwen2.5-1.5B-Instruct | 512 | 0.70 |

The 256-token run was heavily affected by truncation. The 512-token run is the current baseline for future post-training comparisons.

### Run Qwen Baseline

    PYTHONPATH=src python scripts/run_eval.py --config configs/eval_gsm8k_qwen2_5_1_5b_512.yaml

### Inspect Failures

    python scripts/inspect_latest_failures.py --benchmark gsm8k --limit 15
    python scripts/compare_failure_types.py --benchmark gsm8k
    python scripts/compare_runs.py --benchmark gsm8k

## Prepare GSM8K SFT Data

Generate chat-format JSONL training/validation data from GSM8K (no GPU needed, runs anywhere):

    PYTHONPATH=src python scripts/prepare_gsm8k_sft_data.py --config configs/data_gsm8k_sft_small.yaml

This writes:

    data/sft/gsm8k_train_small.jsonl
    data/sft/gsm8k_val_small.jsonl

Both are drawn only from GSM8K's `train` split at disjoint row ranges — the `test` split (used for the baseline above) is never touched, so it stays available for an unbiased base-vs-SFT comparison later. See `docs/dataset_format.md` for the record schema.

## Run LoRA SFT Training

Requires the RunPod GPU (not the Mac). On RunPod, with the venv activated:

    python -m pip install peft trl
    PYTHONPATH=src python scripts/train_sft_lora.py --config configs/train_sft_qwen2_5_1_5b_gsm8k.yaml

This trains a LoRA adapter on top of `Qwen/Qwen2.5-1.5B-Instruct` using the prepared GSM8K SFT data, and saves it to `outputs/sft/qwen2_5_1_5b_gsm8k_lora` (local-only, `outputs/` is gitignored).

### Current Result

| Epoch | eval_loss | eval_mean_token_accuracy |
|---:|---:|---:|
| 1 | 0.3808 | 0.8822 |
| 2 | 0.4005 | 0.8758 |
| 3 | 0.4545 | 0.8707 |

Eval loss rises after epoch 1 (overfitting on the small 200-example set), so the trainer is configured to automatically keep the best (epoch 1) checkpoint rather than the last one.

## Evaluate the SFT Adapter (Base vs SFT)

Same eval, same 100 test examples, only difference is `adapter_path`:

    PYTHONPATH=src python scripts/run_eval.py --config configs/eval_gsm8k_qwen2_5_1_5b_sft.yaml

### Current Result

| | Baseline | SFT adapter |
|---|---:|---:|
| accuracy | 0.70 | 0.45 |
| format_violation | 18 | 3 |
| wrong_numeric_answer | 12 | 52 |

SFT fixed formatting but regressed reasoning accuracy overall. Diagnosed across four rounds (overfitting fix, data scale-up + gentler LoRA, precision control, full-dataset scale-up):

| Experiment | Train examples | Eval dtype | Accuracy |
|---|---:|---|---:|
| Baseline | - | fp16/bf16 | 0.70 |
| v1 | 200 | fp16 | 0.45 |
| v2 | 1500 | bf16 | 0.55 |
| v3 | 7000 | bf16 | 0.57 |

Data quantity clearly hit diminishing returns (1500->7000 examples only gained +2pts) -- more GSM8K data alone is unlikely to close the remaining gap. See `DECISIONS.md` (Decision 020) for the full diagnostic history.

## SFT on the Base Model (headline result)

Since fine-tuning the already-tuned Instruct model only ever regressed accuracy, the next attempt fine-tuned `Qwen/Qwen2.5-1.5B` (base, non-instruct) instead -- a model with no existing GSM8K ability to overwrite:

    PYTHONPATH=src python scripts/prepare_gsm8k_sft_data.py --config configs/data_gsm8k_sft_full.yaml
    PYTHONPATH=src python scripts/train_sft_lora.py --config configs/train_sft_qwen2_5_1_5b_base_gsm8k.yaml
    PYTHONPATH=src python scripts/run_eval.py --config configs/eval_gsm8k_qwen2_5_1_5b_base_sft.yaml

### Result

Full diagnostic path (all four runs):

| Run | Correct | no_numeric_answer | format_violation | wrong_numeric_answer | Accuracy |
|---|---:|---:|---:|---:|---:|
| Base zero-shot | 3 | 25 | 0 | 72 | 0.03 |
| Base zero-shot + reppen | 0 | 76 | 23 | 1 | 0.00 |
| Base + SFT + reppen | 1 | 14 | 53 | 32 | 0.01 |
| **Base + SFT, no reppen** | **37** | 10 | 1 | 52 | **0.37** |

A real, dramatic, qualitative improvement -- from a model that doesn't attempt the task at all (degenerates into repeating a single junk token) to one that reliably formats answers and mostly reasons correctly. See `DECISIONS.md` (Decision 021) for the full diagnostic path, including two real bugs found and fixed along the way (a PEFT/tied-embeddings crash, and a repetition-penalty setting that was accidentally sabotaging the fine-tuned model's eval).

### Next Step

Both SFT tracks are documented (Instruct: regression, Base: success) -- together they show *when* SFT helps vs. hurts, which is the stronger interview story. Next: DPO (below).

## DPO on the SFT'd Model (Stage 19)

Continues the SFT adapter with preference tuning, using on-policy pairs generated from the SFT'd model's own completions -- for training questions it gets wrong, `chosen` = gold reasoning, `rejected` = its actual wrong completion:

    PYTHONPATH=src python scripts/prepare_gsm8k_dpo_data.py --config configs/data_gsm8k_dpo.yaml
    PYTHONPATH=src python scripts/train_dpo.py --config configs/train_dpo_qwen2_5_1_5b_gsm8k.yaml
    PYTHONPATH=src python scripts/run_eval.py --config configs/eval_gsm8k_qwen2_5_1_5b_base_dpo.yaml

### Result

| Run | Correct | no_numeric_answer | format_violation | wrong_numeric_answer | Accuracy |
|---|---:|---:|---:|---:|---:|
| Base zero-shot | 3 | 25 | 0 | 72 | 0.03 |
| Base + SFT | 37 | 10 | 1 | 52 | 0.37 |
| **Base + SFT + DPO** | **51** | 10 | 3 | 36 | **0.51** |

A real, controlled +14-point improvement, concentrated exactly where the preference data targeted it: `wrong_numeric_answer` dropped 52 -> 36 (fixed genuine close-but-wrong reasoning). `no_numeric_answer` (occasional generation collapse) was untouched -- an honest, specific result rather than a blanket improvement. See `DECISIONS.md` (Decision 022) for the full detail, including a training-dynamics note: unlike every SFT run, `eval_loss` decreased monotonically across all 3 epochs with no overfitting.

### Next Step

The core post-training arc (baseline -> SFT -> DPO, 0.03 -> 0.37 -> 0.51) is complete and documented end to end. Next: LLM-as-judge evaluation (below).

## LLM-as-Judge Evaluation

Pairwise comparison using Claude as a judge, instead of exact-match string parsing -- reads two eval runs' `results.csv` files and asks Claude which candidate's reasoning is better for each matching question. Runs entirely on the Mac, no GPU needed.

    export ANTHROPIC_API_KEY=sk-ant-...   # or put it in .env (see .env.example)
    PYTHONPATH=src python scripts/run_llm_judge.py \
      --run-a path/to/sft_run/results.csv --label-a "SFT" \
      --run-b path/to/dpo_run/results.csv --label-b "SFT+DPO" \
      --limit 30 \
      --output reports/judge_sft_vs_dpo.csv

Judge model is Claude Opus 4.8 by default (`--model` to override) -- see `DECISIONS.md` (Decision 023) for the cost/quality reasoning.

### Next Step

Consider serving/inference comparison (vLLM/TensorRT-LLM), or synthetic/self-distilled data generation to push GSM8K accuracy further.
