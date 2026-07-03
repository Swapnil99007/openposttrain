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
