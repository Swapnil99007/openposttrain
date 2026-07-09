
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

### Final Controlled Result (bf16 vs bf16)

Baseline (no adapter) scores **identically** in fp16 and bf16 (0.70 both) -- the base model is numerically robust to precision. The adapter is not (0.49 fp16 -> 0.55 bf16) -- itself a symptom that the fine-tuned model is more numerically fragile than the base model, consistent with everything else observed.

| | Baseline (bf16) | SFT v2 adapter (bf16) |
|---|---:|---:|
| accuracy | 0.70 | 0.55 |

A real, controlled **15-point regression**. Smaller than the uncontrolled fp16 comparison suggested (which conflated a real precision confound with the adapter's actual effect), but still a genuine regression -- more data and gentler LoRA reduced the damage (70->45 uncontrolled, then 70->55 controlled) but did not close the gap.

### v3 Result (full GSM8K train set, ~7000 examples)
Accuracy: **0.57** -- only +2 points over v2's 0.55, despite 4.67x more training data.

| Experiment | Train examples | LoRA r/alpha/lr | Eval dtype | Accuracy |
|---|---:|---|---|---:|
| Baseline (no adapter) | - | - | fp16 or bf16 | 0.70 |
| v1 | 200 | 16/32/2e-4 | fp16 | 0.45 |
| v2 | 1500 | 8/16/1e-4 | fp16 | 0.49 |
| v2 | 1500 | 8/16/1e-4 | bf16 | 0.55 |
| v3 | 7000 | 8/16/1e-4 | bf16 | 0.57 |

Data quantity shows clear diminishing returns: 200->1500 (+4pts), then 1500->7000, a much bigger jump, (+2pts). The precision fix alone (+6pts) mattered more than 5500 additional training examples. Data quantity is very unlikely to close the remaining ~13-point gap on its own.

### Status
Accepted as a documented negative result for the Instruct-model track. Superseded as the project's headline SFT result by Decision 021 (SFT on the base model instead) -- see below.

## Decision 021: SFT on the base (non-instruct) model succeeds where Instruct-model SFT regressed

### Decision
Fine-tune `Qwen/Qwen2.5-1.5B` (base, non-instruct) instead of `-Instruct`, reusing the existing ~7000-example GSM8K SFT data. This is the project's headline SFT result.

### Reason
Decision 020 established that fine-tuning the already-instruction-tuned Instruct model, even after 3 rounds of fixes (more data, gentler LoRA, precision control), only ever regressed accuracy (70% -> 55-57%) -- consistent with SFT overwriting existing tuned behavior rather than teaching something new. The base model has no such behavior to overwrite: its zero-shot accuracy is functionally ~0% (see below), so there's genuine headroom for SFT to teach a real, missing skill.

### What Went Wrong Along the Way (and how each was diagnosed, not guessed)
1. **Base model zero-shot baseline was 0.03, but the raw completions showed the model doesn't attempt the problem at all** -- it echoes the prompt back, then degenerates into repeating a single junk token (`afone`, `aload`) or looping the whole prompt. The "3 correct" and the false `format_violation=0` reading were both artifacts: our evaluator's format check just searches for the substring `"final answer"`, which the model trivially satisfies by echoing the prompt (which contains that literal phrase).
2. **Tested whether this was a decoding artifact** (like the max_new_tokens truncation in Decision 017): added `repetition_penalty`/`no_repeat_ngram_size` to `HFModel.generate()`. Result: accuracy dropped further to 0.00, with even shorter, still-degenerate completions across foreign scripts and junk tokens. Confirmed there's no hidden capability being masked -- the base model genuinely cannot do this task zero-shot, in any decoding configuration tested.
3. **Training crashed twice** trying to borrow the Instruct model's chat template (`chat_template_path`) so the base tokenizer could format conversations -- an unresolved PEFT/TRL interaction between newly-added tokens and tied embeddings (`AttributeError: 'TrainableTokensWrapper' object has no attribute 'bias'`; PEFT's `modules_to_save` fix, applied per TRL's own suggested fix, didn't resolve it -- see huggingface/peft#2777). Sidestepped by training on **plain text prompt/completion** (no chat template, no new tokens at all) instead -- also a better match for how `HFModel.generate()` already evaluates this tokenizer.
4. **Training completed cleanly** (healthy loss curve, same mild overfitting shape as the Instruct-model runs, `load_best_model_at_end` picked epoch 1), but eval with the same `repetition_penalty` settings as the baseline gave only 0.01 -- and inspecting completions showed severe generation collapse (Chinese-script repetition spirals, random JavaScript snippets, alphabet-mashing). Hypothesis: `repetition_penalty`/`no_repeat_ngram_size`, tuned specifically to break the *raw* model's degenerate loop, actively sabotage the *trained* model, which legitimately needs to reuse tokens (numbers, operators, "Final Answer:") during correct reasoning. Confirmed by re-running without them.

### Final Result

All four runs in the diagnostic path, for the full picture:

| Run | Correct | no_numeric_answer | format_violation | wrong_numeric_answer | Accuracy |
|---|---:|---:|---:|---:|---:|
| Base zero-shot | 3 | 25 | 0 | 72 | 0.03 |
| Base zero-shot + reppen | 0 | 76 | 23 | 1 | 0.00 |
| Base + SFT + reppen | 1 | 14 | 53 | 32 | 0.01 |
| **Base + SFT, no reppen** | **37** | 10 | 1 | 52 | **0.37** |

The headline comparison (raw model vs. final result):

| | Raw base model (zero-shot) | Base + SFT |
|---|---:|---:|
| accuracy | 0.03 (functionally ~0 -- see above) | **0.37** |
| correct | 3 | 37 |
| no_numeric_answer (degenerate loop) | 25 | 10 |
| format_violation | 0 (false reading, see above) | 1 |
| wrong_numeric_answer | 72 (mostly noise, see above) | 52 (mostly genuine single-step arithmetic slips) |

A real, dramatic, well-diagnosed improvement -- not just a higher number, but a qualitative change from "the model doesn't attempt the task" to "the model reliably formats answers and mostly reasons correctly, occasionally making an understandable arithmetic mistake." This is the SFT success story the project set out to demonstrate, achieved by fixing the actual root cause identified in Decision 020 (already-tuned model + narrow data = regression) rather than continuing to chase data quantity/quality fixes on the wrong base model.

### Status
Accepted.

## Decision 022: DPO on top of the SFT'd base model improves accuracy further (0.37 -> 0.51)

### Decision
Continue the SFT LoRA adapter with DPO (Stage 19), using on-policy preference pairs generated from the SFT'd model's own completions (see `src/openposttrain/data/gsm8k_dpo.py`): for GSM8K training questions where the model's own greedy completion was wrong, `chosen` = the gold reasoning already prepared for SFT, `rejected` = the model's actual wrong completion.

### Data
300 training + 50 validation candidate questions (reused from the SFT training pool, not the held-out `test` split -- Decision 018 discipline maintained). After filtering to wrong-only: **199 train pairs, 31 validation pairs** (~65% of candidates were wrong, consistent with the ~63% failure rate observed in the SFT eval).

### Training
Continued the existing SFT adapter directly (`AutoPeftModelForCausalLM.from_pretrained(..., is_trainable=True)`, no fresh `peft_config`, per TRL's documented pattern) rather than starting a new adapter -- `beta=0.1`, `learning_rate=1e-5`. Notably, `eval_loss` decreased monotonically across all 3 epochs (0.2398 -> 0.1421 -> 0.1116), unlike every SFT run, which all showed overfitting (eval_loss rising after epoch 1). `rewards/accuracies` climbed to ~97-100%, meaning the model almost perfectly learned to prefer chosen over rejected on this dataset -- worth noting some of that separation is "easy" (some rejected samples are pure degenerate garbage, e.g. one is `"ystatechangeuser"` repeated ~180 times, trivially distinguishable from fluent correct reasoning), not just the harder case of two fluent answers where only one is right.

### Result

**Corrected (see Decision 024): the table below originally compared DPO against the wrong SFT run.** The DPO adapter was continued from a *different* SFT training run than the one independently reported as 0.37 -- the lineage-correct comparison uses that adapter's own eval (0.32):

| Run | Correct | no_numeric_answer | format_violation | wrong_numeric_answer | Accuracy |
|---|---:|---:|---:|---:|---:|
| Base zero-shot | 3 | 25 | 0 | 72 | 0.03 |
| Base + SFT (actual DPO ancestor) | 32 | 17 | 2 | 49 | 0.32 |
| **Base + SFT + DPO** | **51** | 10 | 3 | 36 | **0.51** |

Same eval settings as the SFT result (no repetition_penalty, bf16, same 100 test questions) -- a directly controlled comparison (Decision 012). Inspected completions: clean, correct, well-structured step-by-step reasoning, not gamed formatting.

A real, controlled **+19-point** improvement (0.32 -> 0.51). With the correct lineage, DPO improved on *both* fronts, not just one: `wrong_numeric_answer` dropped 49 -> 36 (fixing genuine close-but-wrong reasoning) **and** `no_numeric_answer` dropped 17 -> 10 (fewer degenerate-loop generations too). The original writeup claimed DPO left the degenerate-loop rate untouched, based on a coincidental match (both runs showed `no_numeric_answer=10`) against the wrong SFT comparison point -- that claim is retracted; DPO's effect was broader than first reported. `format_violation` ticked up slightly (2 -> 3), within noise.

### Status
Accepted. This is now the project's best result across the full pipeline: raw base model (0.03) -> SFT (0.32) -> SFT + DPO (0.51). See Decision 024 for the training-non-determinism finding that necessitated this correction, and for the separately-valid 0.37 SFT data point from an independent training run.

## Decision 024: SFT training is not bit-reproducible across runs, even with a fixed seed (0.37 vs 0.32)

### Finding
When a fresh RunPod pod required retraining the SFT adapter from scratch (same code, same data, same `seed=42`), the retrained adapter evaluated to **0.32** accuracy, not the original **0.37** -- a real 5-point swing between two nominally identical experiments. Both evals used identical settings (no `repetition_penalty`, bf16, greedy `temperature=0.0`, same 100 test questions), so the difference isn't measurement noise -- it's a genuinely different adapter.

The two training runs' `eval_loss` curves are close but not identical:

| Run | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---:|---:|---:|
| Original | 0.4214 | 0.4276 | 0.4577 |
| Retrain | 0.4215 | 0.4272 | 0.4574 |

### Root Cause
GPU training is not bit-reproducible run-to-run even with a fixed seed -- certain CUDA operations (attention, some parallel reductions) are non-deterministic by default (`torch.use_deterministic_algorithms` was never set). The tiny resulting weight differences are enough to flip several borderline greedy-decoding outputs on the 100-question eval set.

### Practical Consequence
The DPO adapter (Decision 022, final result 0.51) was continued from the **retrained** adapter (the one that evaluates to 0.32), not the original 0.37 adapter -- the original pod (and its results) no longer existed by the time DPO training ran. This means the previously-reported "0.37 -> 0.51" comparison mixed two different SFT adapters. The lineage-correct comparison for the actual DPO adapter is:

| | SFT (this adapter's own accuracy) | SFT + DPO (continued from this adapter) |
|---|---:|---:|
| accuracy | 0.32 | 0.51 |

A +19-point improvement from DPO on this specific lineage -- an even stronger result than the originally-reported +14 points, and the correct one to cite going forward. The original 0.37 remains a valid, independently-verified data point in its own right; it just isn't the ancestor of the 0.51 adapter.

### Takeaway
A single point-estimate accuracy from one training run carries real run-to-run variance (here, ~5 points on a 100-example eval) that isn't visible unless you happen to retrain and re-measure. Neither number is "wrong" -- but citing an exact single decimal (e.g. "0.37") as if it were a fixed property of "the SFT approach" overstates precision. If this project needed a more statistically rigorous accuracy claim, the right fix would be averaging over multiple training seeds and/or a larger eval set, not just running once. Given project scope/time, that's noted as a known limitation rather than fixed.

### Status
Accepted as a documented limitation. Both SFT numbers (0.32, 0.37) and the DPO number (0.51) are reported transparently; 0.32 -> 0.51 is used as the primary lineage-correct comparison going forward.

## Decision 023: LLM-as-judge pairwise evaluation, using Claude Opus 4.8

### Decision
Add `src/openposttrain/judge/` + `scripts/run_llm_judge.py`: a pairwise LLM-as-judge comparator that reads two eval runs' `results.csv` files, sends each matching question pair (question, gold solution, candidate A, candidate B) to Claude, and gets back a structured `{winner, reasoning_quality_score, explanation}` verdict via `client.messages.parse(output_format=JudgeVerdict)`.

### Reason
This is a genuinely different evaluation methodology from the project's exact-match evaluator (which reads for a specific number in a specific spot, and which Decision 021 already showed has real brittleness -- the base-model prompt-echo bug). An LLM judge reads for meaning: it can assess reasoning quality even when the string-matching heuristic would misfire, and produces pairwise comparisons (not just per-run accuracy). It's also literally one of the named skills on the Liquid AI job application that motivated starting this project.

### Model Choice
Claude Opus 4.8. Estimated cost per comparison (~800 input / ~150 output tokens): even 100 comparisons is well under $1 with Opus, so at this project's scale, judgment quality mattered more than the cost delta vs. Haiku/Sonnet -- a deliberate choice, not a default, since downgrading model choice for cost savings that don't materially matter isn't a good tradeoff to make silently.

### Design Notes
- Reuses existing `results.csv` outputs directly -- no new generation needed, runs entirely on the Mac, no RunPod/GPU required.
- Retries on judge failure (up to 2x) rather than silently dropping a malformed verdict from the aggregate.
- Pairs rows by index across the two CSVs, with a question-text equality check as a safety guard against accidentally comparing mismatched eval runs.

### Status
Accepted.

## Decision 025: LLM-as-judge result -- SFT+DPO preferred on reasoning quality

### Decision
Ran `scripts/run_llm_judge.py` on the lineage-correct pair: SFT (0.32 accuracy, `20260707_055109` run) vs. SFT+DPO (0.51 accuracy, `20260707_044505` run), 30 question pairs, Claude Opus 4.8 as judge.

### Result
SFT+DPO won 13/30 (43.3%), SFT won 5/30 (16.7%), tie 12/30 (40.0%).

### Interpretation
The judge's preference for SFT+DPO (13 vs. 5, ~2.6x) is directionally consistent with the exact-match accuracy gap (0.32 -> 0.51), obtained via a completely independent method -- reading reasoning quality rather than string-matching a final number. This corroborates that the DPO improvement is real and not an artifact of the exact-match evaluator's specific parsing rules. The high tie rate (40%) is expected: many GSM8K problems are short enough that both models produce equally clear correct (or equally clear wrong) reasoning, leaving little for a judge to differentiate.

### Status
Accepted. Full per-question verdicts saved to `reports/judge_sft_vs_dpo.csv`.

## Decision 026: Add GRPO (RL) stage, continuing the DPO adapter

### Decision
Add `src/openposttrain/data/gsm8k_grpo.py`, `src/openposttrain/training/grpo.py`, `src/openposttrain/training/grpo_rewards.py`, `scripts/prepare_gsm8k_grpo_data.py`, `scripts/train_grpo.py` -- a GRPO (Group Relative Policy Optimization) stage that continues the SFT+DPO adapter with online RL, using TRL's `GRPOTrainer`.

### Reason
SFT and DPO in this project are both offline: the model is trained against a fixed dataset built once ahead of time, never generating anything live during training. GRPO is structurally different -- the model generates a completion during training, a reward function scores it immediately, and the policy updates from that live score, repeating in a loop. This "environment + grader + reward signal" loop is the specific pattern named across OpenAI's Agent Post-Training postings and Anthropic's Research Engineer, Post-Training role (RLHF/RLAIF, training environments, graders), and is the one capability area completely missing from the project before this stage.

### Design
- **Reward functions** (`grpo_rewards.py`): `accuracy_reward` and `format_reward` reuse `extract_model_answer` from the existing GSM8K evaluator (Decision 021) directly -- no new grading logic written. `accuracy_reward` compares the extracted number to `ground_truth`; `format_reward` checks for the "Final Answer:" marker the evaluator already expects. Combined via `reward_weights: [0.8, 0.2]` (accuracy weighted higher; format is a shaping signal).
- **Data** (`gsm8k_grpo.py`, `configs/data_gsm8k_grpo.yaml`): only needs `(prompt, ground_truth)` pairs, not gold reasoning text, since reward is computed from the model's own live generation. Drawn from GSM8K's `train` split rows 350-900 -- still `train`, never `test` (Decision 018), but a different slice than DPO's pairs (rows 0-350 of the same pool) to avoid pure re-training on exactly what DPO already used.
- **Trainer** (`grpo.py`, `train_grpo.py`): continues the existing DPO adapter via `AutoPeftModelForCausalLM.from_pretrained(dpo_adapter_path, is_trainable=True)`, the same continuation pattern DPO used on top of SFT.
- **First-pass config** (conservative, single RTX 3090): `use_vllm=False` (avoid colocate GPU-memory-sharing complexity on the first run), `num_generations=4`, `per_device_train_batch_size=4` (== num_generations, so one prompt's generations exactly fill a device batch), `learning_rate=1e-6` (low -- refining an already-tuned policy, not training from scratch), nonzero `beta=0.02` (TRL's default is 0.0 = no KL regularization, appropriate for training from scratch; a small pull back toward the DPO policy seemed safer here).
- Sourced the current `GRPOTrainer`/`GRPOConfig` API from TRL's docs before writing code (same practice as the DPO stage) rather than assuming a remembered signature, since this API has had real churn (e.g. `loss_type` default changed to `"dapo"`, `scale_rewards` default changed to `"group"`).

### Status
Code written and reward functions sanity-checked locally (pure-Python, no GPU needed). Not yet run -- training itself needs to happen on RunPod. Batch-size/num_generations settings may need adjusting once we see real memory/speed behavior, consistent with how SFT and DPO configs were refined after their first real runs.

## Decision 027: GRPO result -- flat against DPO, likely underpowered config

### Decision
Ran `configs/train_grpo_qwen2_5_1_5b_gsm8k.yaml` (500 train prompts, `num_generations=4`, `learning_rate=1e-6`, `beta=0.02`, 1 epoch), continuing the DPO adapter. Evaluated with the standard GSM8K test-split protocol (same 100 examples, no repetition_penalty) via `configs/eval_gsm8k_qwen2_5_1_5b_base_grpo.yaml`.

### Result
| Run | Correct | no_numeric_answer | format_violation | wrong_numeric_answer | Accuracy |
|---|---:|---:|---:|---:|---:|
| Base + SFT | 32 | 17 | 2 | 49 | 0.32 |
| Base + SFT + DPO | 51 | 10 | 3 | 36 | 0.51 |
| Base + SFT + DPO + GRPO | 52 | 11 | 1 | 36 | 0.52 |

### Interpretation
`wrong_numeric_answer` is identical between DPO and GRPO (36 -> 36) -- the specific failure mode DPO improved didn't move further. The net +1 "correct" is a wash between `format_violation` dropping (3 -> 1) and `no_numeric_answer` rising (10 -> 11), not a consistent signal. Combined with the run-to-run noise already established for this 100-example eval (~5 points, Decision 024), **this result is statistically flat against DPO, not a demonstrated improvement.**

This isn't a failure of the RL mechanism itself -- training metrics were healthy throughout (`format_reward` pinned at 1.0 every step, KL small and stable at ~0.0007-0.0015, no reward collapse or hacking). More likely the config was underpowered to move accuracy further on top of an already-strong DPO policy: only 500 train prompts, a deliberately conservative `learning_rate=1e-6`, and a single epoch. Training and eval logs are the artifact worth keeping regardless -- they demonstrate the actual online generate-grade-update loop working correctly end to end (the specific capability gap Decision 026 set out to close), even though this particular tuning didn't clear the noise floor.

### Status
Accepted as a documented, honestly-reported null result. Next iteration (if pursued) would try a higher learning rate, more train prompts, and/or more epochs -- not pursued further yet given RunPod time/cost.

## Decision 028: GRPO v2 (10x LR, 3x epochs) -- identical result to v1, GRPO stage concluded

### Decision
Ran `configs/train_grpo_qwen2_5_1_5b_gsm8k_v2.yaml`: same 500 train prompts as v1, but `learning_rate` raised 1e-6 -> 1e-5 (10x) and `num_train_epochs` raised 1 -> 3, changing both levers at once as a deliberate speed/rigor trade-off (fewer RunPod round trips, at the cost of not knowing which lever would matter if it worked -- see Decision 027 follow-up discussion).

### Result
| Run | Correct | no_numeric_answer | format_violation | wrong_numeric_answer | Accuracy |
|---|---:|---:|---:|---:|---:|
| Base + SFT + DPO | 51 | 10 | 3 | 36 | 0.51 |
| Base + SFT + DPO + GRPO v1 | 52 | 11 | 1 | 36 | 0.52 |
| Base + SFT + DPO + GRPO v2 | 52 | 11 | 1 | 36 | 0.52 |

v2's failure-type breakdown is **identical** to v1's, not just the same accuracy -- every category matches exactly. This despite real, visible differences in training dynamics: v2's KL stayed an order of magnitude larger throughout (~0.003-0.01 vs. v1's ~0.0007-0.0015, i.e. genuinely more policy movement), and the in-training validation-set reward climbed further (`eval_rewards/accuracy_reward/mean` 0.52 -> 0.58 from v1 to v2). None of that additional movement showed up on the held-out 100-example test split.

### Interpretation
Two conclusions, both worth taking at face value rather than explaining away:
1. **The in-training val-set reward is not a reliable proxy for the held-out benchmark here.** v2 looked meaningfully better by its own validation metric but didn't move the actual eval at all -- a caution against reading training curves as if they were the real result.
2. **This DPO-then-GRPO recipe has plateaued on this eval**, at least across the range of learning rates and epoch counts tried. More aggressive tuning within the same shape of config (same 500 prompts, same reward functions, just higher LR/more epochs) is unlikely to be the lever that moves this further -- a different data source, different reward shaping, or a genuinely different intervention would be needed, not more of the same knob-turning.

### Status
Concluding the GRPO stage here rather than running a v3. Two consistent data points (v1, v2) at 0.52 is a more useful, honest result than a third run chasing a number that two independent configs already agree isn't moving. Full GRPO stage summary: baseline 0.03 -> SFT 0.32 -> DPO 0.51 -> GRPO 0.52, with GRPO's contribution honestly reported as within-noise rather than a demonstrated further gain.
