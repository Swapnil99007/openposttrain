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

## Run Comparison

The project includes a lightweight leaderboard comparison script.

File:

- scripts/compare_runs.py

Purpose:

- Read results/leaderboard.csv
- Filter by benchmark
- Compare models, prompts, decoding settings, and accuracy

## Failure-Type Comparison

The project includes a script to aggregate failure categories across evaluation runs.

File:

- scripts/compare_failure_types.py

Purpose:

- Read each run's results.csv
- Count failure_type values
- Compare model and prompt behavior across runs

## Clean Prompt Comparison

Prompt comparison should use the same model, benchmark, limit, and decoding settings.

Only the prompt_path should change.

This allows fair comparison between prompt versions.

## Prompt Comparison Script

The project includes a prompt comparison script.

File:

- scripts/compare_prompts.py

Purpose:

- Filter leaderboard rows by benchmark and model
- Compare prompt_path values
- Aggregate failure categories for each prompt version

## Experiment Report Generation

The project includes a markdown report generator.

File:

- scripts/generate_experiment_report.py

Purpose:

- Summarize leaderboard runs
- Aggregate failure categories
- Produce a shareable experiment report under reports/

## Real Baseline Evaluation Stage

Before starting post-training, OpenPostTrain runs a meaningful baseline with a real instruction model.

Current baseline model:

- `Qwen/Qwen2.5-1.5B-Instruct`

Current benchmark:

- GSM8K
- test split
- first 100 examples
- deterministic generation
- `temperature=0.0`
- `top_p=1.0`
- `max_new_tokens=512`

### Why this stage exists

Post-training should not be done blindly.

The baseline identifies:
- whether the model can follow the prompt
- whether failures are caused by truncation
- whether failures are caused by evaluator bugs
- whether failures are true reasoning mistakes
- what kind of SFT data should be prepared

### Current result

The current meaningful baseline is:

| Model | Benchmark | Limit | Max New Tokens | Accuracy |
|---|---|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | GSM8K | 100 | 512 | 0.70 |

### Failure Categories

The evaluator tracks:

- `correct`
- `no_numeric_answer`
- `format_violation`
- `wrong_numeric_answer`

After the 512-token Qwen run, most remaining failures are real math/reasoning failures or final-answer-format issues.

### Next Design Step

Done — see "SFT Data Preparation" below. The next stage after that is LoRA SFT training.

## SFT Data Preparation

Before training, GSM8K is converted into chat-format JSONL examples suitable for SFT.

Files:

- `src/openposttrain/data/gsm8k_sft.py` — cleans GSM8K gold solutions (strips calculator annotations like `<<16-3=13>>13`) and builds `{"messages": [...]}` records using the same prompt template the evaluator uses.
- `src/openposttrain/utils/jsonl.py` — generic JSONL writer.
- `configs/data_gsm8k_sft_small.yaml` — dataset split/row-range and output path settings.
- `scripts/prepare_gsm8k_sft_data.py` — CLI that runs the above and writes train/validation JSONL files.
- `docs/dataset_format.md` — documents the record schema and split-hygiene rationale.

Purpose:

- Produce training data whose assistant-turn format (`Final Answer: <number>`) matches exactly what `extract_model_answer` in `src/openposttrain/evals/gsm8k.py` parses, so eval and training are aligned.
- Keep GSM8K's `test` split completely untouched during data prep (see Decision 018), so it stays valid for an unbiased base-vs-SFT comparison later.

Current output:

- `data/sft/gsm8k_train_small.jsonl` — 200 examples (`train[0:200]`)
- `data/sft/gsm8k_val_small.jsonl` — 50 examples (`train[200:250]`)

This data is local-only (`data/` is gitignored) and regenerated by re-running the script.

### Next Design Step (updated)

Done -- see "LoRA SFT Training" below. Next stage is base-vs-SFT evaluation.

## LoRA SFT Training

Files:

- `src/openposttrain/training/sft_lora.py` -- converts our `{"messages": [...]}` records into TRL's "prompt-completion" format (`{"prompt": [...], "completion": [...]}`), builds the `LoraConfig` (PEFT) and `SFTConfig` (TRL).
- `scripts/train_sft_lora.py` -- CLI: loads config, builds dataset/LoRA config/training args, runs `SFTTrainer`, saves the adapter.
- `configs/train_sft_qwen2_5_1_5b_gsm8k.yaml` -- model, data paths, LoRA hyperparameters, training hyperparameters.

Key design choices:

- **Prompt-completion format over `assistant_only_loss`**: TRL's `assistant_only_loss=True` masks non-assistant tokens but requires the chat template to include `{% generation %}` markers, which TRL only guarantees auto-patching for certain model families. Using the "prompt-completion" dataset format gets completion-only loss masking by default, with no chat-template dependency.
- **No QLoRA/4-bit quantization**: Qwen2.5-1.5B is ~3GB in bf16; the RTX 3090's 24GB is more than enough for full-precision-base + LoRA without quantization overhead.
- **`load_best_model_at_end=True`** (Decision 019): the first training run showed clear overfitting past epoch 1 (eval_loss rising while train_loss kept falling) on the small 200-example dataset. Rather than guess a safe epoch count, the trainer now automatically keeps the checkpoint with the lowest eval_loss.

### Current result

3 epochs, 200 train / 50 validation examples, `Qwen/Qwen2.5-1.5B-Instruct`, RunPod RTX 3090:

| Epoch | eval_loss | eval_mean_token_accuracy |
|---:|---:|---:|
| 1 | 0.3808 | 0.8822 |
| 2 | 0.4005 | 0.8758 |
| 3 | 0.4545 | 0.8707 |

The saved adapter (`outputs/sft/qwen2_5_1_5b_gsm8k_lora`) corresponds to epoch 1 (verified via `md5sum` against the per-epoch checkpoints).

Important caveat: `mean_token_accuracy` is a token-level training metric, not GSM8K exact-match accuracy. Whether this adapter actually improves benchmark performance is still unknown until Stage 17 (base-vs-SFT eval).

### Next Design Step

Done -- see "Base vs SFT Evaluation" below.

## Base vs SFT Evaluation

`HFModel` (`src/openposttrain/models/hf_model.py`) takes an optional `adapter_path` and wraps the base model with `PeftModel.from_pretrained` when set. `scripts/run_eval.py` threads this through from the model config. `configs/eval_gsm8k_qwen2_5_1_5b_sft.yaml` mirrors the baseline config exactly except for `adapter_path`, isolating the adapter as the only variable (Decision 012).

### Current result

| | Baseline | SFT adapter |
|---|---:|---:|
| accuracy | 0.70 | 0.45 |
| format_violation | 18 | 3 |
| wrong_numeric_answer | 12 | 52 |

SFT improved formatting compliance but substantially regressed actual reasoning accuracy -- a net accuracy drop. See Decision 020 for the full analysis and root-cause hypothesis (200-example training set too small/narrow).

### Next Design Step

Done -- four rounds of diagnosis completed: overfitting fix (Decision 019), data scale-up + gentler LoRA (v2), eval dtype control (fp16 vs bf16), and full-dataset scale-up (v3, ~7000 examples). Results across all experiments:

| Experiment | Train examples | Eval dtype | Accuracy |
|---|---:|---|---:|
| Baseline | - | fp16/bf16 | 0.70 |
| v1 | 200 | fp16 | 0.45 |
| v2 | 1500 | bf16 | 0.55 |
| v3 | 7000 | bf16 | 0.57 |

Data quantity shows clear diminishing returns (1500->7000 only gained +2pts), meaning quantity alone is unlikely to close the remaining ~13-point gap. `HFModel` now supports an optional `dtype` override (`float16`/`bfloat16`/`float32`) for precision-control experiments like this one.

Open decision: try regenerating SFT targets from the base model's own verified-correct reasoning instead of GSM8K's terse gold text, vs. move forward to other pipeline stages and treat this as a documented finding to revisit later.

## Base-Model SFT (headline result)

Decision: fine-tune `Qwen/Qwen2.5-1.5B` (base, non-instruct) instead of `-Instruct`, since the Instruct-model track (above) only ever regressed accuracy -- consistent with SFT overwriting existing tuned behavior rather than teaching something new. The base model has no such behavior to overwrite.

Files/changes beyond what's listed above:

- `src/openposttrain/training/sft_lora.py` -- added `to_plain_prompt_completion()` and a `conversational` flag on `load_sft_dataset()`. Training uses plain-text prompt/completion (not chat messages), avoiding a chat-template/new-token/tied-embeddings PEFT bug hit twice when trying `chat_template_path` (unresolved upstream, huggingface/peft#2777).
- `src/openposttrain/models/hf_model.py` -- added `repetition_penalty`/`no_repeat_ngram_size` (opt-in, `None` by default) to fix/diagnose base-model decoding degeneration; also fixed tokenizer loading to prefer `adapter_path` over `model_name` when present.
- `configs/train_sft_qwen2_5_1_5b_base_gsm8k.yaml`, `configs/eval_gsm8k_qwen2_5_1_5b_base*.yaml` -- training and eval configs for this track.

Key lesson: **decoding settings tuned to fix one model's failure mode can actively harm a different model.** `repetition_penalty=1.3`/`no_repeat_ngram_size=3` were necessary to reveal that the raw base model has no real zero-shot capability (breaks a degenerate repetition loop), but the same settings, applied "for a controlled comparison," severely damaged the fine-tuned model's legitimate generation (forcing it away from tokens it correctly needed to reuse). Removing them for the SFT'd model's eval was the correct call, verified by inspecting completions, not assumed.

### Final Result

| | Raw base model (zero-shot) | Base + SFT |
|---|---:|---:|
| accuracy | 0.03 (functionally ~0 -- see Decision 021) | **0.37** |

A real, qualitative improvement: from "doesn't attempt the task" (degenerate repetition) to "reliably formats answers, mostly reasons correctly." Full diagnostic writeup in Decision 021.

### Next Design Step

Both SFT tracks are now documented (Instruct: regression, Decision 020; Base: real success, Decision 021), together demonstrating *when* SFT helps vs. hurts. Next candidates: Stage 19 (DPO) or synthetic/self-distilled data generation.
