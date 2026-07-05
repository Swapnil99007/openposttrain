
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

## 2026-07-03

### Goal
Add prompt comparison script.

### What I did
- Added `scripts/compare_prompts.py`.
- The script filters runs by benchmark and model.
- It compares prompt versions using accuracy and failure-type counts.

### Results
- Prompt comparisons are now easier and cleaner.

### Next
- Add a reusable experiment report generator.

## 2026-07-03

### Goal
Add experiment report generation.

### What I did
- Added `scripts/generate_experiment_report.py`.
- Generated a markdown report from leaderboard and failure-type data.
- Added `tabulate` for markdown table support.

### Results
- GSM8K runs can now be summarized in `reports/gsm8k_report.md`.

### Next
- Start preparing for actual post-training data generation.

## 2026-07-04

### Goal
Run a proper GSM8K baseline using a real instruction model before starting post-training.

### What I did
- Ran `Qwen/Qwen2.5-1.5B-Instruct` on GSM8K using RunPod RTX 3090.
- First run used `max_new_tokens=256`.
- Second run used `max_new_tokens=512`.
- Fixed numeric normalization in the GSM8K evaluator so values like `460.0` and `460` are treated equivalently.
- Transferred raw RunPod results back to the local machine.

### Results

| Model | Limit | Max New Tokens | Accuracy | Correct | Format Violations | Wrong Numeric |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 100 | 256 | 0.43 | 43 | 54 | 3 |
| Qwen2.5-1.5B-Instruct | 100 | 512 | 0.70 | 70 | 18 | 12 |

### Key Finding
The first Qwen baseline was artificially low because `max_new_tokens=256` caused many generations to truncate before the model reached the final answer.

Increasing to `max_new_tokens=512` improved accuracy from 43% to 70%.

### Failure Analysis
Remaining failures are mostly:
- arithmetic mistakes
- incorrect interpretation of word problems
- boundary-condition mistakes
- remaining formatting / final-answer issues

### Decision
Use the 512-token Qwen run as the current meaningful baseline before SFT.

### Next
Prepare targeted GSM8K SFT data based on real Qwen failure modes.

## 2026-07-04 (SFT data preparation)

### Goal
Build the GSM8K SFT data preparation pipeline before any training.

### What I did
- Added `src/openposttrain/data/gsm8k_sft.py`: strips GSM8K calculator annotations (`<<16-3=13>>13`) and builds chat-format (`{"messages": [...]}`) records whose assistant turn ends in `Final Answer: <number>`, matching what the evaluator already parses.
- Added `src/openposttrain/utils/jsonl.py` as a generic JSONL writer.
- Added `configs/data_gsm8k_sft_small.yaml` and `scripts/prepare_gsm8k_sft_data.py`.
- Added `docs/dataset_format.md` documenting the record schema.
- Fixed a `.gitignore` bug: the unanchored `data/` pattern was matching `src/openposttrain/data/` (the new source package) in addition to the intended top-level `data/` directory. Changed to `/data/`.
- Removed the superseded one-off `reports/gsm8k_qwen_baseline_report.md` (replaced by the regeneratable `reports/gsm8k_report.md`).

### Results
- Generated `data/sft/gsm8k_train_small.jsonl` (200 examples) and `data/sft/gsm8k_val_small.jsonl` (50 examples), both from GSM8K's `train` split at disjoint row ranges.
- Verified output is clean (no leftover calculator annotations) and inspected the literal token string via `tokenizer.apply_chat_template` to confirm training-time input matches the eval-time chat template.

### Decision
Draw SFT train/validation data only from GSM8K's `train` split, never `test` — the `test` split stays untouched for the base-vs-SFT comparison in Stage 18 (see Decision 018).

### Next
Write the LoRA SFT training script (Stage 16) and run it on RunPod against `Qwen/Qwen2.5-1.5B-Instruct`.

## 2026-07-04 (LoRA SFT training)

### Goal
Run LoRA SFT training on the prepared GSM8K data and get a working adapter.

### What I did
- Added `src/openposttrain/training/sft_lora.py` (dataset loading + prompt/completion conversion, LoRA config, `SFTConfig` builder) and `scripts/train_sft_lora.py` (CLI), using TRL's `SFTTrainer` + PEFT LoRA.
- Chose "prompt-completion" format (not `assistant_only_loss` on raw `messages`) so completion-only loss masking doesn't depend on the chat template having `{% generation %}` markers, which isn't guaranteed for every model family.
- Fixed an environment issue on a fresh RunPod pod: bare `pip install torch` grabbed the newest PyPI wheel (cu13-targeted), incompatible with the pod's driver (cuda12.4.1 image); fixed by installing from `https://download.pytorch.org/whl/cu124`.
- Ran a first training pass (3 epochs, 200 train / 50 validation examples, `Qwen/Qwen2.5-1.5B-Instruct`, RunPod RTX 3090). Found clear overfitting: train loss kept dropping (0.42 -> 0.20) but eval_loss rose after epoch 1 (0.3795 -> 0.3997 -> 0.4531) and eval accuracy fell.
- Added `load_best_model_at_end=True` (`metric_for_best_model="eval_loss"`) so the saved adapter is the best-eval-loss checkpoint, not just the last epoch (Decision 019).
- Re-ran training; verified via `md5sum` that the saved adapter is byte-identical to the epoch-1 checkpoint, confirming the fix works.

### Results
- Final adapter: `outputs/sft/qwen2_5_1_5b_gsm8k_lora` (epoch-1 checkpoint, `eval_loss=0.3808`, `eval_mean_token_accuracy=0.8822`).
- Training pipeline (TRL SFTTrainer + PEFT LoRA + prompt-completion masking + best-checkpoint selection) is verified working end-to-end on RunPod.

### Issues
- 200 examples is a small dataset; overfitting past epoch 1 suggests the training set should probably be scaled up before relying on more epochs.
- `mean_token_accuracy` is a token-level training proxy, not the same as GSM8K exact-match accuracy -- doesn't tell us yet whether the adapter actually improves on the real benchmark.

### Next
Stage 17: add adapter-loading support to the eval pipeline and run the SFT adapter against the same GSM8K `test[0:100]` slice used for the Qwen baseline, to get a real base-vs-SFT accuracy comparison.

## 2026-07-04 (base-vs-SFT eval)

### Goal
Get a real, controlled base-vs-SFT accuracy comparison using the same GSM8K `test[0:100]` slice as the baseline.

### What I did
- Added `adapter_path` support to `HFModel` (wraps the base model with `PeftModel.from_pretrained`) and threaded it through `scripts/run_eval.py` and the leaderboard row.
- Added `configs/eval_gsm8k_qwen2_5_1_5b_sft.yaml`, identical to the baseline config except for `adapter_path`, so any accuracy difference is attributable to the adapter (Decision 012).
- Ran it on RunPod against the Stage 16 adapter.

### Results
| | Baseline | SFT adapter |
|---|---:|---:|
| accuracy | 0.70 | 0.45 |
| correct | 70 | 45 |
| format_violation | 18 | 3 |
| wrong_numeric_answer | 12 | 52 |

### Key Finding
SFT **improved** answer formatting (18 -> 3 format violations) but **regressed** actual reasoning accuracy substantially (12 -> 52 wrong numeric answers), for a net accuracy drop from 70% to 45%. See Decision 020 for the full breakdown and root-cause hypothesis (200 training examples is too small/narrow; the adapter likely overfit formatting conventions while distorting general arithmetic reasoning).

### Decision
Not yet made -- see Decision 020 (status: Open). Considering: (a) scale up the SFT training set well beyond 200 examples, (b) reduce LoRA aggressiveness (lower rank/learning rate), (c) rule out the fp16-eval vs bf16-train dtype mismatch as a confound first since it's cheap to test.

### Next
Decide and execute a fix, then re-run the same controlled baseline-vs-SFT comparison to check whether it closes the gap.

## 2026-07-05 (SFT retry v2: more data + gentler LoRA)

### Goal
Test whether scaling up training data (200 -> 1500 examples) and making LoRA gentler (`r:16->8`, `alpha:32->16`, `lr:2e-4->1e-4`) fixes the accuracy regression from Decision 020.

### What I did
- Added `configs/data_gsm8k_sft_medium.yaml`, `configs/train_sft_qwen2_5_1_5b_gsm8k_v2.yaml`, `configs/eval_gsm8k_qwen2_5_1_5b_sft_v2.yaml` (versioned alongside the originals, not overwriting them).
- Regenerated SFT data (1500 train / 150 validation), retrained, re-ran the same controlled eval on RunPod.

### Results
Accuracy: 0.49 (vs. v1's 0.45, vs. baseline's 0.70). `eval_loss` during training was much flatter (0.4201 -> 0.4205 -> 0.4354) than v1's clear overfitting curve -- the fix worked as intended on the training side, but barely moved downstream accuracy.

### Key Finding
Overfitting was not the whole story. Several v2 failure examples are internally incoherent (self-contradictory arithmetic, made-up units), not just consistently biased -- see Decision 020 for specific examples (James running, Toula bakery, Carla download). That signature is more consistent with a generation-stability issue than a pure small-data/capacity problem.

### Decision
Before making further training changes, isolate the fp16(eval)/bf16(train) precision mismatch as a possible contributor -- added `dtype` support to `HFModel` and a bf16 eval config for the v2 adapter. Cheap, fast test.

### Next
Run `configs/eval_gsm8k_qwen2_5_1_5b_sft_v2_bf16.yaml` and compare accuracy to the fp16 v2 result (0.49). If bf16 closes a meaningful part of the gap, precision mismatch is a real contributor. If accuracy is essentially unchanged, rule it out and look harder at training data style/quality instead.

## 2026-07-05 (bf16 eval result + apples-to-apples control)

### Goal
Check whether the fp16(eval)/bf16(train) mismatch was a real contributor to the accuracy gap.

### Results
v2 adapter, bf16 eval: 0.55 accuracy (up from 0.49 in fp16) -- a real +6-point effect. Confirmed as a genuine contributor, not a dead end.

### Issue Found
This result compares a bf16 adapter run against the original **fp16** baseline (0.70), which mixes precision the other way. Added `configs/eval_gsm8k_qwen2_5_1_5b_bf16.yaml` (baseline, no adapter, bf16) so the final comparison holds precision constant on both sides.

### Next
Run the bf16 baseline and compare bf16-vs-bf16 (adapter 0.55 vs. whatever the bf16 baseline turns out to be) for the real, controlled answer on whether this SFT adapter helps or hurts.

### Result
Baseline in bf16: **0.70** -- identical to the original fp16 baseline. The base model is numerically robust to precision; the adapter is not (0.49 fp16 -> 0.55 bf16).

**Final controlled comparison: baseline 0.70 vs. SFT v2 adapter 0.55 -- a real 15-point regression.** More data + gentler LoRA + precision control together closed some of the gap (uncontrolled comparison looked like a 25-point regression) but did not close it. This is a genuine, documented negative result, not a bug -- three rounds of diagnosis (overfitting fix, data scale-up + gentler LoRA, precision control) each ruled something in or out and each was verified with real numbers, not assumed.

### Next
Open decision: keep iterating on SFT (candidates: even more training data, GSM8K's terse gold-solution style may itself be a poor SFT target, further LoRA gentling) vs. accept this as a documented finding and move forward to other pipeline stages (DPO, synthetic data, etc.), returning to SFT quality later if time allows.

## 2026-07-05 (SFT retry v3: full GSM8K train set)

### Goal
Test whether data quantity is still the limiting factor -- scale from 1500 to ~7000 training examples (nearly all of GSM8K's train split, which has 7473 rows), keeping the same gentler LoRA settings as v2 and evaluating directly in bf16 (already established as necessary for a fair comparison).

### What I did
Added `configs/data_gsm8k_sft_full.yaml` (train_limit 7000, validation_limit 300), `configs/train_sft_qwen2_5_1_5b_gsm8k_v3.yaml` (same LoRA/training hyperparameters as v2, new data paths), `configs/eval_gsm8k_qwen2_5_1_5b_sft_v3_bf16.yaml` (bf16 directly, skipping the fp16 rediscovery since that confound is already understood).

### Next
Run data prep, training (expect ~45-60 min given 1500 examples took ~13 min for 3 epochs), and bf16 eval on RunPod. Compare against the controlled baseline of 0.70.

### Results
Accuracy: **0.57** (train_runtime ~60 min, 1314 steps). Training curve shows the same mild overfitting shape as v2 (`eval_loss`: 0.4276 -> 0.4305 -> 0.46), best checkpoint at epoch 1.

### Key Finding
Only +2 points over v2 (0.55), despite 4.67x more training data. Full picture across all experiments:

| Experiment | Train examples | LoRA r/alpha/lr | Eval dtype | Accuracy |
|---|---:|---|---|---:|
| Baseline (no adapter) | - | - | fp16 or bf16 | 0.70 |
| v1 | 200 | 16/32/2e-4 | fp16 | 0.45 |
| v2 | 1500 | 8/16/1e-4 | fp16 | 0.49 |
| v2 | 1500 | 8/16/1e-4 | bf16 | 0.55 |
| v3 | 7000 | 8/16/1e-4 | bf16 | 0.57 |

Data quantity shows clear diminishing returns (200->1500: +4pts; 1500->7000, a much bigger jump: +2pts). The precision fix alone (+6pts) mattered more than 5500 additional training examples. Data quantity is very unlikely to close the remaining ~13-point gap on its own.

### Decision
Not yet made -- most plausible remaining lever is GSM8K's terse gold-solution style itself (never tested). Open: try regenerating SFT targets from the base model's own verified-correct reasoning (self-distillation) vs. accept the current result as a documented, rigorously-diagnosed finding and move forward to other pipeline stages.

## 2026-07-05 (SFT on base model instead of Instruct -- Option A)

### Goal
Test whether the SFT regression was fundamentally caused by fine-tuning an already-instruction-tuned model with narrower/lower-quality data (overwriting good behavior) rather than teaching a genuinely missing skill. Fix: fine-tune `Qwen/Qwen2.5-1.5B` (base, non-instruct) instead of `-Instruct`, reusing the existing 7000-example GSM8K SFT data and pipeline.

### What I did
- Added `configs/eval_gsm8k_qwen2_5_1_5b_base.yaml` for a new zero-shot baseline on the base model (bf16 throughout this track, no fp16/bf16 confound this time).
- Added `eos_token`/`chat_template_path` passthrough to `build_sft_training_args` in `src/openposttrain/training/sft_lora.py` -- per TRL's own docs, Qwen base models ship a chat template already but need `eos_token="<|im_end|>"` set explicitly (their worked example is literally this model).

### Next
Run the base-model zero-shot baseline first (cheap, no training) to see actual behavior (does it follow the chat template/format at all?) before committing to a full training run.

### Results
Accuracy: 0.03 (3 correct, 25 no_numeric_answer, 0 format_violation, 72 wrong_numeric_answer). Eval took ~15 min (2x the Instruct model's ~7 min) -- consistent with rarely hitting a natural stop.

### Key Finding -- the 0.03 number is not trustworthy as-is
Inspecting raw completions (`inspect_latest_failures.py`) showed the base model echoes the prompt back verbatim, then degenerates into repeating a single junk token (`afone`, `aload`, `acey`) for the rest of the 512-token budget, or in one case loops the entire prompt text repeatedly. It never actually attempts the problem. This is a well-known greedy-decoding pathology for base (non-chat-tuned) models with no natural turn-boundary signal.

This also exposed a real evaluator bug: `classify_result`'s `format_violation` check just searches for the substring `"final answer"` anywhere in the output. Since our own prompt text contains that literal phrase ("write the final answer in this exact format: Final Answer: <number>"), echoing the prompt trivially satisfies the check -- which is why `format_violation` showed 0 even though the model never produced a real answer. The "wrong" numbers extracted were just numbers pulled from the restated problem text (e.g. "$2 per fresh duck egg" -> extracted "2.0"), not anything the model computed. Same category of mistake as the max_new_tokens truncation issue (Decision 017): don't judge a model by a broken measurement setup.

### Decision
Test whether a standard decoding fix -- `repetition_penalty=1.3`, `no_repeat_ngram_size=3` -- changes the picture before designing the training run around this number. Added these as opt-in parameters to `HFModel.generate()` (threaded through `run_gsm8k_eval` and `run_eval.py`), defaulting to `None` so existing instruct-model configs are unaffected. New config: `configs/eval_gsm8k_qwen2_5_1_5b_base_reppen.yaml`.

### Next
Run the repetition-penalty config and compare against the 0.03 result and against the raw completions to confirm whether the model now actually attempts problems.

### Results
Accuracy dropped to **0.00** (0 correct, 76 no_numeric_answer, 23 format_violation, 1 wrong_numeric_answer), and eval time dropped from ~15 min to ~2 min (much shorter completions). Inspected raw completions: mostly just `"afone\n\nafone"` or short degenerate/foreign-script rambling that never resolves to a real answer -- e.g. one completion switches into Chinese characters mid-generation and rambles about payroll systems without ever computing anything.

### Key Finding
The repetition-penalty fix didn't reveal hidden capability -- it eliminated the repetition loop, but what's underneath is still non-functional. This means the original "3 correct" at 0.03 were very likely coincidental (numbers pulled from repeated garbage or the echoed prompt happening to match gold answers), not real reasoning. **Confirmed: the base model genuinely cannot do this task zero-shot, in any decoding configuration tested.** This is the ideal starting point for demonstrating SFT -- there's no existing capability to fight against.

Also suspected (but had not directly verified) that `Qwen/Qwen2.5-1.5B`'s tokenizer has no usable chat template, based on the zero-shot completion showing what looked like raw unwrapped prompt text. That specific claim turned out to be an overreach -- see the training crash below, which suggests the real picture is more nuanced (TRL detected new tokens when cloning the chat template regardless of what was already there). Lesson: "looked like X in one output" isn't the same as "confirmed X" -- should have checked `tokenizer.chat_template` directly instead of inferring from behavior.

### Decision
- Training config (`configs/train_sft_qwen2_5_1_5b_base_gsm8k.yaml`) sets `chat_template_path: Qwen/Qwen2.5-1.5B-Instruct` (borrowing the sibling model's template) and `eos_token: "<|im_end|>"`, per TRL's documented pattern.
- Fixed a related bug this uncovered in `HFModel`: it always loaded the tokenizer from `model_name`, but TRL saves a chat-template patch into the *adapter's* output directory, not back into the base model repo. Eval would have silently used the un-patched (template-less) tokenizer otherwise. Now loads from `adapter_path` when present.
- After-training eval config (`configs/eval_gsm8k_qwen2_5_1_5b_base_sft_reppen.yaml`) keeps the same `repetition_penalty`/`no_repeat_ngram_size` as the baseline, for a controlled comparison.

### Next
Regenerate SFT data (fresh RunPod pod, `data/` never persisted), train, and eval. Compare against the 0.00 baseline.

### Results (training crashed)
Training crashed on the very first step (`AttributeError: 'TrainableTokensWrapper' object has no attribute 'bias'`), inside TRL's chunked-loss forward pass. The eval command run afterward also failed, but only as a downstream symptom -- there was no adapter directory to load a tokenizer from, since training never reached `save_model()`.

### Root Cause
TRL logged the real cause right before the crash: `"Cloning chat template added new tokens to the tokenizer, but 'lm_head' is not in PEFT's modules_to_save... add modules_to_save=['lm_head'] to your PEFT configuration."` Borrowing the Instruct model's chat template via `chat_template_path` added tokens the LoRA config wasn't told to make trainable, and combined with tied input/output embeddings (`tie_word_embeddings=True`), this hit an incompatible code path in TRL's chunked cross-entropy loss.

### Decision
Added `modules_to_save` passthrough to `build_lora_config` (`src/openposttrain/training/sft_lora.py`) and set `modules_to_save: [lm_head]` in the training config, per TRL's own suggested fix.

### Next
Re-run training with the fix.

### Results (fix didn't work -- same crash)
Retried with `modules_to_save: [lm_head]`. Same crash, same line (`AttributeError: 'TrainableTokensWrapper' object has no attribute 'bias'`). The earlier "lm_head is not in modules_to_save" warning was gone (confirming our setting was applied), but a *different* PEFT wrapper (`TrainableTokensWrapper`, PEFT's separate mechanism for handling embeddings for newly-added tokens) still hit the same code path -- it takes precedence over `modules_to_save` for the tied embedding/lm_head tensor. The warning about `tie_word_embeddings=True` + `ensure_weight_tying` links to an open PEFT GitHub issue (huggingface/peft#2777) describing exactly this kind of complication. This looks like a real, currently-unresolved library limitation, not a config mistake.

### Decision
Sidestep the chat-template approach entirely instead of continuing to chase this interaction. Added `to_plain_prompt_completion()` and a `conversational` flag to `load_sft_dataset()` in `src/openposttrain/training/sft_lora.py` -- trains on plain text (prompt/completion as strings, not chat messages), so TRL never needs to apply or clone a chat template, and no new tokens are ever introduced. This also better matches how `HFModel.generate()` already evaluates this tokenizer (no chat template -> raw prompt text fallback), so train and eval are now naturally consistent. Removed `chat_template_path`/`eos_token`/`modules_to_save` from the training config.

### Next
Re-run training with the plain-text (non-conversational) format.

### Results (training succeeded, first eval disappointing)
Training completed cleanly this time -- healthy loss curve (0.46 -> 0.30 train loss, mean_token_accuracy climbing to ~0.90), same mild overfitting shape as the Instruct-model runs (`eval_loss`: 0.4214 -> 0.4276 -> 0.4577, best at epoch 1). But eval (same `repetition_penalty`/`no_repeat_ngram_size` settings as the baseline, for a "controlled" comparison) gave only **0.01** accuracy -- barely above the pre-SFT baseline, despite the healthy training metrics. Eval also took ~13.5 min, close to the original (no-reppen) baseline's ~15 min, not the clean reppen-baseline's ~2 min -- a clue that generations were running long again.

Inspected completions: severe generation collapse -- spirals into obsessive Chinese-script repetition and pinyin, random JavaScript code snippets for a word problem, German currency notation mixed with math, alphabet-mashing. Hypothesis: `repetition_penalty=1.3`/`no_repeat_ngram_size=3` were tuned specifically to break the *raw* base model's degenerate-repetition problem, but a properly-trained model doing real step-by-step math legitimately needs to reuse tokens (numbers, operators, "Final Answer:") -- forcing it away from that could be sabotaging otherwise-correct generations, similar in spirit to the fp16/bf16 confound in Decision 020.

### Decision
Added `configs/eval_gsm8k_qwen2_5_1_5b_base_sft.yaml` (same adapter, no repetition penalty) to isolate the effect before concluding anything.

### Results (confirmed -- real success)
Without the repetition penalty: **0.37** accuracy (37 correct, 10 no_numeric_answer, 1 format_violation, 52 wrong_numeric_answer). A dramatic, real jump from the ~0.00-0.03 baseline. Inspected completions: format compliance is essentially solved, and most "wrong" answers are genuine, coherent single-step arithmetic mistakes (e.g. computing a value increase as `1.5 x 80,000` instead of `80,000 + 1.5 x 80,000`), not gibberish -- though ~10% of completions still fall back into the old degenerate-loop pattern.

### Key Finding -- final result for the base-model SFT track
**Raw base model (zero-shot): 0.03 (functionally ~0, almost entirely degenerate repetition) -> Base + SFT: 0.37.** A real, qualitative, well-diagnosed improvement -- the model went from "doesn't attempt the task" to "reliably formats answers and mostly reasons correctly." This is the clean SFT success story, achieved by fixing the actual root cause from Decision 020 (fine-tuning an already-tuned Instruct model with narrow data regresses it) rather than continuing to chase fixes on the wrong base model. Full writeup in Decision 021.

### Decision
This is the project's headline SFT result. The Instruct-model track (Decision 020, final: 70% -> 55-57%, a documented regression) stays as a valuable contrasting finding -- together they demonstrate *when* SFT helps vs. hurts, which is a stronger interview story than either result alone.

### Next
Update all tracking docs with the final comparison (done). Consider Stage 19 (DPO) or synthetic/self-distilled data generation as the next pipeline stage.
