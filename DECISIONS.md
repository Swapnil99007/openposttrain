
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
