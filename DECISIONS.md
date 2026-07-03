
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
