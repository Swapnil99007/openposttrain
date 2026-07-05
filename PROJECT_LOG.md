
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
