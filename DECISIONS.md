
## Decision 002: Use YAML configs for evaluation runs

### Decision

Evaluation runs will be configured through YAML files instead of hardcoded script values.

### Reason

This makes experiments reproducible and easier to compare.

A single runner can support multiple models and benchmarks by changing config files.

### Alternatives Considered

- Hardcode model and benchmark settings in each script.
- Use only command-line arguments.
- Use Python config files.

### Tradeoff

YAML configs add slightly more structure upfront, but make the project cleaner as the number of experiments grows.

### Status

Accepted.

## Decision 003: Start with a local CSV leaderboard

### Decision

Evaluation summaries will be appended to a local CSV leaderboard at `results/leaderboard.csv`.

### Reason

A CSV leaderboard is simple, transparent, and easy to inspect while the project is still early.

It gives us experiment comparison without adding MLflow or W&B too early.

### Alternatives Considered

- Store only per-run `summary.json` files.
- Add MLflow immediately.
- Use a database table from the beginning.

### Tradeoff

A CSV file is less powerful than a full experiment tracker, but it keeps the early system simple.

### Status

Accepted.

## Decision 004: Add failure inspection before adding more benchmarks

### Decision

Add a lightweight failure inspection script before expanding to more benchmarks.

### Reason

Aggregate metrics such as accuracy do not show why a model failed.

Inspecting failures helps distinguish between model mistakes, prompt issues, and answer extraction bugs.

### Alternatives Considered

- Only rely on leaderboard metrics.
- Add more benchmarks first.
- Use a dashboard immediately.

### Tradeoff

This adds another small script, but improves debugging and evaluation quality.

### Status

Accepted.

## Decision 005: Add simple GSM8K failure categories

### Decision

Each GSM8K result will include a `failure_type`.

### Reason

Accuracy alone does not explain model behavior.

Failure categories help identify whether the problem is missing numeric output, bad formatting, or wrong reasoning.

### Alternatives Considered

- Keep only `is_correct`.
- Manually inspect all failures.
- Add a full error taxonomy immediately.

### Tradeoff

The categories are simple and heuristic-based, but they are useful enough for early debugging.

### Status

Accepted.

## Decision 006: Store generation parameters in YAML configs

### Decision

Generation settings such as `max_new_tokens`, `temperature`, and `top_p` will be read from YAML configs.

### Reason

Evaluation behavior must be reproducible.

Changing decoding settings can change model outputs and benchmark scores.

### Alternatives Considered

- Hardcode generation settings in evaluator code.
- Pass generation settings only as CLI arguments.

### Tradeoff

YAML config adds small overhead, but makes experiments easier to reproduce and compare.

### Status

Accepted.

## Decision 007: Add latest-run inspection helper

### Decision

Add a script that automatically finds the latest benchmark run and inspects its failures.

### Reason

Manual result paths are long and error-prone.

### Alternatives Considered

- Continue manually copying result paths.
- Add a full dashboard immediately.

### Tradeoff

The helper uses directory name sorting, which is simple but assumes timestamp-prefixed run folders.

### Status

Accepted.

## Decision 008: Add a small instruction-tuned local model

### Decision

Add `HuggingFaceTB/SmolLM2-135M-Instruct` as a local evaluation model.

### Reason

`sshleifer/tiny-gpt2` is useful only for smoke testing, but it cannot meaningfully follow instructions.

A small instruction-tuned model allows local testing of prompt-following, answer formatting, and failure categories without downloading a multi-GB model.

### Alternatives Considered

- Continue using only tiny-gpt2.
- Download Qwen2.5-1.5B locally.
- Move immediately to RunPod.

### Tradeoff

SmolLM2-135M is still too small for strong GSM8K accuracy, but it is useful for local evaluator validation.

### Status

Accepted.

## Decision 009: Version prompt templates as files

### Decision

Prompt templates will be stored as files under `prompts/` and selected from YAML configs.

### Reason

Prompt wording affects model outputs and benchmark results.

Versioning prompts makes evaluation runs more reproducible and easier to compare.

### Alternatives Considered

- Keep prompts hardcoded in evaluator code.
- Store prompts only inside YAML files.
- Pass prompts through command-line arguments.

### Tradeoff

Prompt files add another project artifact, but they make prompt experiments cleaner.

### Status

Accepted.

## Decision 010: Add simple leaderboard comparison script

### Decision

Add a script to compare evaluation runs from the local leaderboard.

### Reason

As more models and prompts are tested, manually reading leaderboard rows becomes inconvenient.

### Status

Accepted.

## Decision 011: Compare failure types across runs

### Decision

Add a script that aggregates failure_type counts across evaluation runs.

### Reason

When accuracy is low, failure categories provide more useful signal than accuracy alone.

### Status

Accepted.

## Decision 012: Compare prompts under controlled settings

### Decision

Prompt comparisons should hold model, benchmark, limit, and decoding settings constant.

### Reason

This isolates prompt wording as the variable being tested.

### Status

Accepted.

## Decision 013: Add prompt comparison script

### Decision

Add a script to compare prompt versions for the same model and benchmark.

### Reason

Prompt experiments should be easy to compare without manually filtering the leaderboard.

### Status

Accepted.

## Decision 014: Generate markdown experiment reports

### Decision

Add markdown report generation from local experiment data.

### Reason

Reports make project progress easier to review, share, and explain in interviews.

### Status

Accepted.

## Decision 016: Run a real instruction-model baseline before SFT

### Decision
Before preparing SFT data, run a meaningful baseline using `Qwen/Qwen2.5-1.5B-Instruct`.

### Reason
Earlier runs with `tiny-gpt2` and `SmolLM2-135M-Instruct` were useful for validating the evaluation pipeline, but they were too weak to guide real post-training decisions.

A stronger instruction model gives a more realistic view of:
- reasoning failures
- formatting failures
- generation-length issues
- evaluator bugs

### Result
The Qwen2.5-1.5B baseline reached 70% accuracy on the first 100 GSM8K test examples with `max_new_tokens=512`.

### Status
Accepted.

## Decision 017: Use 512 max new tokens for GSM8K baseline evaluation

### Decision
Use `max_new_tokens=512` for the current GSM8K baseline instead of `256`.

### Reason
The 256-token run caused many responses to truncate before the final answer. This made the model look worse than it actually was.

### Evidence
Qwen2.5-1.5B-Instruct results:

| Max New Tokens | Accuracy | Format Violations |
|---:|---:|---:|
| 256 | 0.43 | 54 |
| 512 | 0.70 | 18 |

### Tradeoff
512 tokens increases evaluation runtime, but gives a more accurate baseline.

### Status
Accepted.

## Decision 018: Draw SFT train/validation data only from GSM8K's train split

### Decision
`configs/data_gsm8k_sft_small.yaml` builds both the SFT train set and the SFT validation set from GSM8K's `train` split only, using disjoint row ranges (validation starts immediately after where train ends). The `test` split is never read during data prep.

### Reason
The Qwen baseline (Decisions 016/017) used `test[0:100]`, and Stage 18 will re-run that same 100-example slice to compare base vs. SFT accuracy. If SFT validation data were drawn from `test`, validation loss could end up correlated with the exact examples used for the final resume-facing comparison, undermining its credibility.

### Alternatives Considered
- Draw validation from `test` split (rejected: risks overlap with the future base-vs-SFT comparison set).
- Draw validation from a random sample of `train` instead of an index range (rejected for now: an offset range is simpler and just as effective at this data size).

### Status
Accepted.

## Decision 019: Use load_best_model_at_end for LoRA SFT training

### Decision
`build_sft_training_args` in `src/openposttrain/training/sft_lora.py` sets `load_best_model_at_end=True`, `metric_for_best_model="eval_loss"`, `greater_is_better=False`, so the final saved adapter is whichever epoch had the lowest validation loss, not simply the last epoch trained.

### Reason
The first real training run (3 epochs, 200 train / 50 validation examples, `Qwen/Qwen2.5-1.5B-Instruct`) showed train loss decreasing every step (0.42 -> 0.20) while eval_loss increased after epoch 1 (0.3795 -> 0.3997 -> 0.4531) and eval_mean_token_accuracy decreased (0.8822 -> 0.8761 -> 0.8708). This is overfitting on a small dataset: without this setting, the saved adapter would have been the most-overfit epoch-3 checkpoint.

### Alternatives Considered
- Manually pick a fixed lower epoch count (rejected: guessing an epoch count is less defensible than letting eval loss decide, and would need re-tuning if dataset size changes).
- Scale up the training set size instead (may still be worth doing later, but doesn't replace having a correctness guardrail in the training script itself).

### Verification
Re-ran training on RunPod with the fix. Same overfitting shape as before (`eval_loss`: 0.3808 -> 0.4005 -> 0.4545 across epochs 1-3), confirming epoch 1 as best. Verified directly via `md5sum` that the top-level `adapter_model.safetensors` is byte-identical to `checkpoint-13` (epoch 1), not `checkpoint-39` (epoch 3) -- `load_best_model_at_end` is working as intended.

### Status
Accepted.

## Decision 020: 200-example LoRA SFT regresses GSM8K accuracy (70% -> 45%)

### Finding
Running the Stage 16 adapter through the same eval used for the baseline (`configs/eval_gsm8k_qwen2_5_1_5b_sft.yaml`, identical settings to the baseline config except `adapter_path`) gave **0.45 accuracy, down from the 0.70 baseline**.

Failure-type breakdown tells a specific story, not just "worse":

| | Baseline | SFT adapter |
|---|---:|---:|
| correct | 70 | 45 |
| format_violation | 18 | 3 |
| wrong_numeric_answer | 12 | 52 |

Formatting compliance improved sharply (18 -> 3). Actual reasoning got much worse (12 -> 52 wrong numeric answers), including basic arithmetic slips (e.g. example 2: model computes `180 x 2 = 360` instead of `180 x 3 = 540`) that look more elementary than the baseline's failures.

### Hypothesis
200 training examples is too small and narrow. LoRA applied broadly (all attention + MLP projections) at `lr=2e-4` had enough signal to learn the `Final Answer: N` formatting convention, but not enough diversity to reinforce general arithmetic reasoning -- and likely nudged shared weights in ways that hurt the base model's existing math ability. This is consistent with the overfitting already observed in training metrics (Decision 019): same root cause, now confirmed on the real downstream benchmark rather than just the training-loss proxy.

Secondary confound worth ruling out (not expected to explain the full gap): eval loads the base model in fp16, training ran in bf16.

### Status
Retrying with a combined fix: scale training data 200 -> 1500 examples (`configs/data_gsm8k_sft_medium.yaml`) and make LoRA less aggressive -- `r: 16->8`, `alpha: 32->16` (keeping `alpha=2xr`), `learning_rate: 2e-4->1e-4` (`configs/train_sft_qwen2_5_1_5b_gsm8k_v2.yaml`). Both changed at once, so if this works we won't know which lever mattered more -- accepted tradeoff for a faster path to a working adapter. Original 200-example configs kept as-is for reproducibility/comparison.

### v2 Result (more data + gentler LoRA)
Accuracy: **0.49** (49 correct, 1 format_violation, 50 wrong_numeric_answer) -- only marginally better than v1's 0.45, still far below the 0.70 baseline.

Training metrics did behave as intended: `eval_loss` is much flatter across epochs (0.4201 -> 0.4205 -> 0.4354) vs. v1's clear overfitting shape (0.3808 -> 0.3997 -> 0.4531). So the fix genuinely reduced overfitting during training -- but that barely moved downstream accuracy. **Overfitting is not the whole story.**

Failure examples show something more specific than "worse reasoning": several are internally incoherent, not just wrong. E.g. James running: model writes "He runs 540 meters a week because 540 x 3 = 1620" (self-contradictory -- states 540 then multiplies it by 3 anyway). Toula bakery: computes `12*68`, `12*80`, `12*55` for all three items -- using "12" (the literal meaning of "dozen") as the multiplier instead of the actual quantity purchased (3, 2, 6 dozen), a systematic misapplied heuristic. Carla download: introduces "hours" out of nowhere in a GB/minutes problem. This looks like generation instability, not simply biased-but-coherent errors -- which is a different signature than plain small-data overfitting.

### Next
Isolate the fp16(eval)/bf16(train) precision mismatch as a possible contributor before making further training changes -- added `dtype` support to `HFModel`/`run_eval.py` and `configs/eval_gsm8k_qwen2_5_1_5b_sft_v2_bf16.yaml` (same v2 adapter, eval in bf16 instead of fp16). Cheap to test, and the incoherent-output pattern is consistent with a numerics artifact compounding over long greedy-decoded generations, not just with a data/capacity problem.

### bf16 Eval Result -- confirmed, but not the whole gap
v2 adapter, bf16 eval: **0.55 accuracy**, up from 0.49 in fp16 -- a real +6-point effect from matching eval precision to training precision. Confirms the mismatch was a genuine contributor.

But this now compares a bf16 adapter run against the original **fp16** baseline (0.70) -- mixing precisions the other way. Added `configs/eval_gsm8k_qwen2_5_1_5b_bf16.yaml` (baseline, no adapter, bf16) to get a consistent same-precision comparison before drawing conclusions about the adapter's true effect.
