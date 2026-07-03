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
