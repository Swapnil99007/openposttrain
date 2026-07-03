# OpenPostTrain Design

## Goal

OpenPostTrain is an end-to-end LLM post-training platform.

The project is designed to support:

- Benchmark-based model evaluation
- LLM-as-a-judge evaluation
- Synthetic data generation
- Data filtering and curation
- Supervised fine-tuning
- Preference optimization
- Model comparison
- Production-style inference

## Current Architecture

YAML Config
    |
    v
Evaluation Runner
    |
    v
Model Wrapper
    |
    v
Benchmark Evaluator
    |
    v
Metrics + Result Files

## Components

### 1. YAML Configs

Configs define model, benchmark, generation, and output settings.

Current config:

- configs/eval_gsm8k_tiny.yaml

### 2. Model Wrapper

File:

- src/openposttrain/models/hf_model.py

Purpose:

- Load HuggingFace causal language models
- Support chat-template and non-chat-template models
- Generate model responses from prompts

### 3. Benchmark Evaluator

File:

- src/openposttrain/evals/gsm8k.py

Purpose:

- Load GSM8K
- Build math prompts
- Run model generation
- Extract numeric answers
- Compute exact-match accuracy

### 4. Evaluation Runner

File:

- scripts/run_eval.py

Purpose:

- Load YAML config
- Create model
- Choose benchmark evaluator
- Save summary and per-example results

## Current Limitations

- Only GSM8K is supported.
- Local smoke tests use sshleifer/tiny-gpt2.
- Real model evaluation will be moved to RunPod or external model cache.

## Leaderboard Tracking

The evaluation runner appends each completed evaluation run to a local leaderboard file.

Default path:

- results/leaderboard.csv

Purpose:

- Track model performance across runs
- Compare models and benchmarks
- Provide a lightweight experiment history before adding MLflow or W&B

Current leaderboard fields:

- timestamp
- run_name
- model_name
- benchmark
- split
- limit
- accuracy
- num_examples
- config_path
- output_dir

The leaderboard is currently stored under results/ and is not committed to GitHub.

## Failure Inspection

The project includes a result inspection script for analyzing incorrect examples.

File:

- scripts/inspect_failures.py

Purpose:

- Load a per-run results.csv file
- Count passed and failed examples
- Print failed examples in a readable format
- Help diagnose model failures, prompt issues, and answer extraction errors

This is important because aggregate metrics alone do not explain why a model failed.

## GSM8K Failure Categorization

The GSM8K evaluator assigns a failure type to each example.

Current categories:

- correct
- no_numeric_answer
- format_violation
- wrong_numeric_answer

Purpose:

- Separate reasoning failures from formatting failures
- Diagnose answer extraction problems
- Improve evaluation quality beyond aggregate accuracy

## Generation Configuration

Generation settings are read from YAML configs and passed into model generation.

Current supported generation fields:

- max_new_tokens
- temperature
- top_p

Purpose:

- Make evaluation runs reproducible
- Avoid hardcoded decoding behavior
- Compare how decoding settings affect benchmark performance

For deterministic evaluation, temperature is usually set to 0.0.

## Latest Run Inspection

The project includes a helper script to inspect the most recent run for a benchmark.

File:

- scripts/inspect_latest_failures.py

Purpose:

- Avoid manually copying long results paths
- Find the latest benchmark run automatically
- Reuse the standard failure inspection script

## Local Instruction Model Evaluation

The project now supports evaluating a small instruction-tuned local model.

Config:

- configs/eval_gsm8k_smollm2_135m.yaml

Model:

- HuggingFaceTB/SmolLM2-135M-Instruct

Purpose:

- Move beyond smoke testing with tiny-gpt2
- Test whether an instruction-tuned model follows the GSM8K prompt format
- Compare failure behavior between non-instruction and instruction-tuned models

## Prompt Template Versioning

Prompt templates are stored as separate files under `prompts/`.

Current GSM8K prompt templates:

- prompts/gsm8k_v1.txt
- prompts/gsm8k_v2_strict.txt

Purpose:

- Avoid hardcoding prompts inside evaluator code
- Make prompt changes reproducible
- Compare model behavior across prompt versions
- Track whether stricter prompts reduce formatting failures

Evaluation configs specify the prompt path using:

    prompt_path: prompts/gsm8k_v2_strict.txt
