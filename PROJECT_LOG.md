
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
