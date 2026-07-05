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

SFT fixed formatting but regressed reasoning accuracy overall. After scaling up training data (200 -> 1500 examples), using gentler LoRA settings, and controlling for an fp16-eval/bf16-train precision mismatch (base model is precision-robust, 0.70 either way; the adapter is not), the final controlled comparison is:

| | Baseline (bf16) | SFT adapter (bf16) |
|---|---:|---:|
| accuracy | 0.70 | 0.55 |

A real 15-point regression that survived three rounds of fixes. See `DECISIONS.md` (Decision 020) for the full diagnostic history.

### Next Step

Open decision: keep iterating on SFT data quality/quantity, or move forward to other pipeline stages and revisit later.
