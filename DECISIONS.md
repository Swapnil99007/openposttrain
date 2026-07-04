
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
