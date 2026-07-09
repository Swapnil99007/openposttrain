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

Note: a later independent retrain of this same SFT recipe evaluated to 0.32, not 0.37, despite identical settings and seed -- GPU training isn't bit-reproducible run-to-run. See Decision 024.

### Next Step

Both SFT tracks are documented (Instruct: regression, Base: success) -- together they show *when* SFT helps vs. hurts, which is the stronger interview story. Next: DPO (below).

## DPO on the SFT'd Model (Stage 19)

Continues the SFT adapter with preference tuning, using on-policy pairs generated from the SFT'd model's own completions -- for training questions it gets wrong, `chosen` = gold reasoning, `rejected` = its actual wrong completion:

    PYTHONPATH=src python scripts/prepare_gsm8k_dpo_data.py --config configs/data_gsm8k_dpo.yaml
    PYTHONPATH=src python scripts/train_dpo.py --config configs/train_dpo_qwen2_5_1_5b_gsm8k.yaml
    PYTHONPATH=src python scripts/run_eval.py --config configs/eval_gsm8k_qwen2_5_1_5b_base_dpo.yaml

### Result

The DPO adapter was continued from a specific SFT adapter -- its own eval accuracy (0.32, not the separately-run 0.37; see Decision 024) is the correct "before" for this comparison:

| Run | Correct | no_numeric_answer | format_violation | wrong_numeric_answer | Accuracy |
|---|---:|---:|---:|---:|---:|
| Base zero-shot | 3 | 25 | 0 | 72 | 0.03 |
| Base + SFT (actual DPO ancestor) | 32 | 17 | 2 | 49 | 0.32 |
| **Base + SFT + DPO** | **51** | 10 | 3 | 36 | **0.51** |

A real, controlled **+19-point** improvement (0.32 -> 0.51), on *both* fronts: `wrong_numeric_answer` dropped 49 -> 36 (fixed genuine close-but-wrong reasoning) and `no_numeric_answer` dropped 17 -> 10 (fewer degenerate-loop generations too). See `DECISIONS.md` (Decision 022) for the full detail, including a training-dynamics note: unlike every SFT run, `eval_loss` decreased monotonically across all 3 epochs with no overfitting.

![GSM8K accuracy across the post-training arc](docs/plots/accuracy_progression.png)

![Failure-type breakdown across the post-training arc](docs/plots/failure_types.png)

*(Charts include GRPO, covered below -- regenerate with `PYTHONPATH=src python scripts/plot_results.py`.)*

### Next Step

The core post-training arc (baseline -> SFT -> DPO, 0.03 -> 0.32 -> 0.51) is complete and documented end to end. Next: LLM-as-judge evaluation (below).

## LLM-as-Judge Evaluation

Pairwise comparison using Claude as a judge, instead of exact-match string parsing -- reads two eval runs' `results.csv` files and asks Claude which candidate's reasoning is better for each matching question. Runs entirely on the Mac, no GPU needed.

    export ANTHROPIC_API_KEY=sk-ant-...   # or put it in .env (see .env.example)
    PYTHONPATH=src python scripts/run_llm_judge.py \
      --run-a path/to/sft_run/results.csv --label-a "SFT" \
      --run-b path/to/dpo_run/results.csv --label-b "SFT+DPO" \
      --limit 30 \
      --output reports/judge_sft_vs_dpo.csv

Judge model is Claude Opus 4.8 by default (`--model` to override) -- see `DECISIONS.md` (Decision 023) for the cost/quality reasoning.

### Result

30 SFT vs. SFT+DPO pairs judged:

| Winner | Count | Rate |
|---|---:|---:|
| SFT+DPO | 13 | 43.3% |
| SFT | 5 | 16.7% |
| tie | 12 | 40.0% |

![LLM-as-judge win rate: SFT vs SFT+DPO](docs/plots/judge_win_rate.png)

Confirms the exact-match accuracy gain (0.32 -> 0.51) qualitatively: DPO wins on reasoning quality far more often than it loses, judged independently of whether the final number happens to match. See `DECISIONS.md` (Decision 025).

### Next Step

Continue to GRPO (below), then the serving/inference comparison further down.

## GRPO (RL)

SFT and DPO above are both offline: trained against a fixed dataset built once ahead of time. GRPO is online RL -- the model generates a completion live during training, a reward function grades it immediately, and the policy updates from that score. Continues the DPO adapter; reward functions reuse the existing GSM8K evaluator's answer-extraction logic directly rather than new grading code. See `DECISIONS.md` (Decision 026) for the full design.

    PYTHONPATH=src python scripts/prepare_gsm8k_grpo_data.py --config configs/data_gsm8k_grpo.yaml
    PYTHONPATH=src python scripts/train_grpo.py --config configs/train_grpo_qwen2_5_1_5b_gsm8k.yaml

### Result

Two configs tried -- v1 (conservative: `lr=1e-6`, 1 epoch) and v2 (`lr=1e-5`, 3 epochs) -- both continuing the same DPO adapter over the same 500 training prompts:

| Run | Correct | no_numeric_answer | format_violation | wrong_numeric_answer | Accuracy |
|---|---:|---:|---:|---:|---:|
| Base + SFT + DPO | 51 | 10 | 3 | 36 | 0.51 |
| Base + SFT + DPO + GRPO v1 | 52 | 11 | 1 | 36 | 0.52 |
| Base + SFT + DPO + GRPO v2 | 52 | 11 | 1 | 36 | 0.52 |

v1's +1 point is within the ~5-point run-to-run noise already established for this eval (see DPO's non-determinism finding above) -- statistically flat, not an improvement. v2 (10x the learning rate, 3x the epochs) landed on the **exact same failure-type breakdown as v1**, despite genuinely more policy movement during training (KL an order of magnitude larger, in-training validation reward climbing to 0.58 vs. v1's 0.52) -- that extra movement didn't transfer to the held-out benchmark at all. Read together, these two runs are stronger evidence of a real plateau than either alone: training itself works correctly end-to-end (the online generate-grade-update loop, stable reward/KL, no collapse in either run) -- this DPO-then-GRPO recipe just isn't moving this particular eval further within the LR/epoch range tried. Full analysis in `DECISIONS.md` (Decisions 027-028).

## Serving/Inference Comparison (vLLM vs. HF Transformers)

A different competency than the training stages above: how the same trained adapter behaves under a production-oriented serving stack instead of the naive HF Transformers loop used for every eval so far.

    PYTHONPATH=src python scripts/run_eval_vllm.py --config configs/eval_gsm8k_qwen2_5_1_5b_base_dpo_vllm.yaml

### Result

Same DPO adapter, same 100 GSM8K test questions, same greedy-decoding parameters, same evaluator:

| Backend | Accuracy | Wall clock (100 examples) |
|---|---:|---:|
| HF Transformers (sequential) | 0.51 | ~11 min |
| vLLM (single batched call) | 0.65 | ~3 sec |

**Throughput**: ~250x, the expected result of continuous batching vs. a one-request-at-a-time loop.

**Accuracy**: the +14-point gap was not expected, and is well outside this eval's established ~5-point noise band. Investigated rather than reported at face value -- ruled out a tokenizer mismatch (real bug, fixed, didn't change the result), then diffed all 100 completions between backends. Finding: vLLM produced zero degenerate/malformed completions on this eval, while HF Transformers produced 13 (including the exact repetition-loop failure mode from the base-model SFT track above -- e.g. one HF completion was the token `afone` repeated ~90 times, where vLLM's completion for the identical question was a clean, correct solution). Where vLLM *was* wrong, it was a genuine reasoning slip, not garbage. Same weights, same LoRA adapter, same greedy decoding -- but different attention/LoRA kernels between the two backends produce numerically different logits, and greedy decoding's autoregressive nature means any early divergence cascades through the whole completion. This is the same non-determinism principle as the GPU training finding above, now shown to apply at inference time between serving stacks too. Full analysis in `DECISIONS.md` (Decisions 029-030).

### Next Step

Both findings (throughput and correctness) are documented as final for this stage.
