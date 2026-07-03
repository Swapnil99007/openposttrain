
## 2026-07-03

### Goal
Build the first working evaluation pipeline for OpenPostTrain.

### What I did
- Added HuggingFace model wrapper.
- Added GSM8K evaluator.
- Added runner script for local evaluation.
- Switched from Qwen2.5-1.5B to sshleifer/tiny-gpt2 due to local disk limits.
- Fixed GSM8K loading by using openai/gsm8k.

### Results
- Model: sshleifer/tiny-gpt2
- Benchmark: GSM8K
- Split: test
- Limit: 3
- Accuracy: 0.0
- Output directory: results/gsm8k/20260703_145148

### Issues
- Qwen2.5-1.5B is too large for local disk.
- tiny-gpt2 is only useful for smoke testing, not real accuracy.

### Next
- Add YAML config loading so model and benchmark settings are not hardcoded.
- Add leaderboard.csv to compare runs.
- Later run real models on RunPod or external model cache.

## 2026-07-03

### Goal
Add a simple leaderboard for evaluation runs.

### What I did
- Added `src/openposttrain/utils/leaderboard.py`.
- Updated `scripts/run_eval.py` to append each run summary to `results/leaderboard.csv`.
- Added `leaderboard_path` to the GSM8K tiny config.
- Documented leaderboard tracking in DESIGN.md.
- Added a decision record for using a local CSV leaderboard.
- Updated README with leaderboard behavior.

### Results
- Each evaluation run now saves:
  - per-run `summary.json`
  - per-run `results.csv`
  - global `results/leaderboard.csv`

### Issues
- Leaderboard is local-only because `results/` is ignored by Git.

### Next
- Add a reusable result inspection script to print wrong examples.

## 2026-07-03

### Goal
Add a failure inspection utility for evaluation results.

### What I did
- Added `scripts/inspect_failures.py`.
- The script reads a per-run `results.csv`.
- It reports total examples, passed examples, failed examples, and accuracy.
- It prints failed examples with question, gold answer, extracted model answer, and full model response.

### Results
- Can now inspect model failures from any evaluation run.

### Issues
- tiny-gpt2 failures are expected and not meaningful for quality analysis.

### Next
- Add cleaner error categorization for GSM8K failures.

## 2026-07-03

### Goal
Add simple GSM8K failure categorization.

### What I did
- Added `failure_type` to GSM8K evaluation results.
- Updated `run_eval.py` to write failure types into results.csv.
- Updated `inspect_failures.py` to show failure type counts.
- Fixed empty extracted answers showing as pandas NaN during inspection.

### Results
- Failure inspection now shows both aggregate accuracy and failure categories.

### Issues
- Current categories are heuristic-based and may need refinement for stronger models.

### Next
- Add configurable generation parameters from YAML into model.generate.

## 2026-07-03

### Goal
Make generation parameters configurable from YAML.

### What I did
- Updated GSM8K evaluator to accept generation settings.
- Updated run_eval.py to read max_new_tokens, temperature, and top_p from YAML.
- Added generation settings to run summaries and leaderboard rows.

### Results
- Evaluation runs now record decoding parameters.
- Future model comparisons will be more reproducible.

### Issues
- tiny-gpt2 is still only a smoke-test model.

### Next
- Add an automatic latest-run helper or result inspection by run directory.

## 2026-07-03

### Goal
Add a helper for inspecting the latest evaluation run.

### What I did
- Added `scripts/inspect_latest_failures.py`.
- The script finds the latest run directory for a benchmark.
- It reuses `inspect_failures.py` to print failed examples.

### Results
- No need to manually copy long results.csv paths.

### Issues
- Latest run is determined by sorted directory name.

### Next
- Add support for running a slightly better tiny instruction model or API-based model wrapper.

## 2026-07-03

### Goal
Add a better local instruction-tuned model for GSM8K evaluation.

### What I did
- Added `configs/eval_gsm8k_smollm2_135m.yaml`.
- Ran GSM8K evaluation with `HuggingFaceTB/SmolLM2-135M-Instruct`.
- Compared the run against the earlier tiny-gpt2 smoke test using `results/leaderboard.csv`.
- Inspected failure examples with the latest-run helper.

### Results
- Added a local instruction-tuned model path for evaluation.
- Leaderboard can now compare tiny-gpt2 and SmolLM2 runs.

### Issues
- SmolLM2-135M is still a small model and may not achieve high GSM8K accuracy.

### Next
- Improve prompts and add prompt template versioning.

## 2026-07-03

### Goal
Add prompt template versioning.

### What I did
- Added prompt templates under `prompts/`.
- Added prompt loading utility.
- Updated GSM8K evaluator to use external prompt templates.
- Updated evaluation configs to include `prompt_path`.
- Ran SmolLM2 with the stricter GSM8K prompt.

### Results
- Prompt versions are now tracked explicitly.
- Evaluation summaries and leaderboard rows include the prompt path.

### Issues
- Prompt improvements may reduce formatting failures but cannot fix weak reasoning in very small models.

### Next
- Compare prompt versions more systematically using the leaderboard and failure types.

## 2026-07-03

### Goal
Add a simple run comparison utility.

### What I did
- Added `scripts/compare_runs.py`.
- The script reads `results/leaderboard.csv`.
- It can filter runs by benchmark.

### Results
- Evaluation runs can now be compared from one command.

### Next
- Add failure-type aggregation across runs.

## 2026-07-03

### Goal
Add failure-type comparison across runs.

### What I did
- Added `scripts/compare_failure_types.py`.
- The script reads the leaderboard.
- For each run, it loads the corresponding results.csv.
- It aggregates failure_type counts.

### Results
- Can now compare not only accuracy, but also failure behavior across models and prompts.

### Next
- Add a cleaner prompt comparison experiment with v1 vs v2 on the same model.

## 2026-07-03

### Goal
Run clean prompt comparison for SmolLM2.

### What I did
- Added `configs/eval_gsm8k_smollm2_135m_v1.yaml`.
- Ran SmolLM2 with GSM8K prompt v1.
- Compared prompt v1 and prompt v2 strict using failure-type aggregation.

### Results
- Can now compare prompt versions with explicit prompt_path tracking.

### Next
- Add a prompt comparison summary script filtered by model.
