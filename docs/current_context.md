# Current Project Context

## Project
OpenPostTrain is an end-to-end LLM post-training project.

The goal is to build practical experience with:

- evaluation
- failure analysis
- SFT data preparation
- LoRA supervised fine-tuning
- base vs fine-tuned model comparison
- later preference tuning / DPO

## Current Stage
The full post-training arc is done: baseline -> SFT -> DPO -> GRPO, with the base-model track as the headline result (0.03 -> 0.32 -> 0.51 -> 0.52). See below for the complete picture, including the corrected SFT number and the honest GRPO plateau finding. Next candidates: serving/inference comparison, or synthetic/self-distilled data generation.

## Current Baseline

Model:

- `Qwen/Qwen2.5-1.5B-Instruct`

Benchmark:

- GSM8K
- test split
- first 100 examples

Hardware:

- RunPod RTX 3090

Prompt:

- `prompts/gsm8k_v1.txt`

Generation:

- deterministic
- `temperature=0.0`
- `top_p=1.0`
- `max_new_tokens=512`

## Baseline Results

| Run | Max New Tokens | Accuracy | Correct | Format Violations | Wrong Numeric |
|---|---:|---:|---:|---:|---:|
| Qwen baseline | 256 | 0.43 | 43 | 54 | 3 |
| Qwen baseline | 512 | 0.70 | 70 | 18 | 12 |

## Main Finding
The 256-token Qwen run was not a valid final baseline because many generations were truncated.

The 512-token Qwen run is the current meaningful baseline.

## Current Failure Pattern
Remaining failures include:

- arithmetic errors
- word-problem interpretation errors
- boundary condition errors
- final-answer formatting issues

## SFT Data Prepared

GSM8K SFT data generated at three sizes as the post-training investigation scaled up: `data/sft/gsm8k_train_small.jsonl` (200, used for Instruct-track v1), `..._medium.jsonl` (1500, v2), `..._full.jsonl` (7000, v3 and the base-model track). All are chat-format JSONL (`{"messages": [...]}`), assistant turn ending in `Final Answer: <number>` to match the evaluator's parsing. GSM8K's `test` split was never touched by any of these, so it stayed valid for base-vs-SFT comparison throughout. See `docs/dataset_format.md`.

## SFT Track 1: Instruct Model (Decision 020) -- documented regression

Fine-tuning `Qwen/Qwen2.5-1.5B-Instruct` (already heavily instruction-tuned) only ever regressed accuracy, across four rounds of fixes:

| Experiment | Train examples | Eval dtype | Accuracy |
|---|---:|---|---:|
| Baseline (no adapter) | - | fp16/bf16 | 0.70 |
| v1 | 200 | fp16 | 0.45 |
| v2 | 1500 | bf16 | 0.55 |
| v3 | 7000 | bf16 | 0.57 |

Root cause: fine-tuning an already-tuned model with narrower/lower-quality data tends to overwrite good behavior rather than add new capability. Data quantity alone showed clear diminishing returns (1500->7000 only gained +2pts). Full diagnostic history in `DECISIONS.md` Decision 020.

## SFT Track 2: Base Model (Decision 021) -- headline success

Pivoted to fine-tuning `Qwen/Qwen2.5-1.5B` (base, non-instruct) instead -- a model with no existing GSM8K ability to overwrite:

| | Raw base model (zero-shot) | Base + SFT |
|---|---:|---:|
| accuracy | 0.03 (functionally ~0, almost entirely degenerate repetition) | **0.37** |

A real, dramatic, qualitative improvement: the raw model doesn't attempt the task at all (echoes the prompt, then repeats a junk token); the SFT'd model reliably formats answers and mostly reasons correctly, with most "wrong" answers being genuine single-step arithmetic slips rather than gibberish.

Two real bugs found and fixed along the way: (1) borrowing the Instruct model's chat template crashed on an unresolved PEFT/tied-embeddings interaction -- fixed by training on plain text instead of chat-formatted messages; (2) the same `repetition_penalty` settings used to diagnose the raw model's degenerate decoding were, when kept "for a controlled comparison," actively sabotaging the fine-tuned model's legitimate generation -- removing them for the SFT'd model's eval was the correct call, verified by inspecting completions. Full diagnostic path in `DECISIONS.md` Decision 021.

Note: an independent retrain of this exact recipe later evaluated to 0.32, not 0.37, despite an identical seed -- GPU training isn't bit-reproducible run-to-run. See Decision 024.

## Stage 19: DPO on the SFT'd Base Model (Decision 022, corrected) -- further improvement

Continued the SFT LoRA adapter with DPO, using on-policy preference pairs: for training questions the SFT'd model gets wrong, `chosen` = gold reasoning, `rejected` = the model's own actual wrong completion (targets real, current failure modes rather than generic bad answers).

**Correction**: the DPO adapter was continued from a different SFT training run than the one reported as 0.37 above (Decision 024) -- the lineage-correct comparison uses that adapter's own eval (0.32):

| Run | Correct | no_numeric_answer | format_violation | wrong_numeric_answer | Accuracy |
|---|---:|---:|---:|---:|---:|
| Base zero-shot | 3 | 25 | 0 | 72 | 0.03 |
| Base + SFT (actual DPO ancestor) | 32 | 17 | 2 | 49 | 0.32 |
| **Base + SFT + DPO** | **51** | 10 | 3 | 36 | **0.51** |

A real, controlled **+19-point** improvement, on *both* failure modes: `wrong_numeric_answer` dropped 49 -> 36 (fixed close-but-wrong reasoning) and `no_numeric_answer` dropped 17 -> 10 (fewer degenerate-loop generations too) -- broader than the originally-reported "reasoning only" effect, which was an artifact of comparing against the wrong SFT run. Training note: `eval_loss` decreased monotonically across all 3 epochs (no overfitting, unlike every SFT run). Both the SFT and DPO adapters were transferred off the ephemeral RunPod pod to the Mac (`outputs/sft/`, `outputs/dpo/`, both gitignored) so future pods can re-upload rather than retrain.

## LLM-as-Judge Confirmation (Decision 025)

Ran the pairwise judge (Claude Opus 4.8) on the SFT (0.32) vs. SFT+DPO (0.51) run, 30 questions: **SFT+DPO won 13/30 (43.3%), SFT won 5/30 (16.7%), tie 12/30 (40.0%)**. An independent, reasoning-quality-based method (not string-matching a number) confirms the same direction as the exact-match accuracy gap -- the DPO improvement isn't an artifact of the evaluator's parsing rules. Per-question verdicts in `reports/judge_sft_vs_dpo.csv`.

## Stage 20: GRPO (RL) -- concluded, plateau at 0.52 (Decisions 026-028)

SFT and DPO are both offline (trained against a fixed dataset built once ahead of time). GRPO is online RL: the model generates a completion live during training, a reward function scores it immediately, and the policy updates from that score. This closes a real gap -- the OpenAI Agent Post-Training and Anthropic Post-Training job postings this project is aimed at all center on exactly this loop (environments, graders, reward signals), which nothing built so far demonstrates.

Design: continues the DPO adapter via `AutoPeftModelForCausalLM`, reward functions reuse the existing GSM8K evaluator's `extract_model_answer` directly (no new grading code), data is fresh `(prompt, ground_truth)` pairs from GSM8K train rows 350-900.

Ran twice on RunPod (several environment issues along the way both times -- numpy/Python-version mismatch, a torch pin silently reverting the cu124 install, `trl` needing a newer torch than initially installed, stale torchvision/torchaudio breaking the import chain, and pip's resolver re-clobbering torch a second time via an unpinned transitive dependency -- all resolved, see PROJECT_LOG for the full sequence):

- **v1** (500 prompts, `lr=1e-6`, 1 epoch): trained cleanly, landed at 0.52 vs. DPO's 0.51 -- within the ~5-point run-to-run noise already established for this eval (Decision 024), so statistically flat.
- **v2** (`lr=1e-5`, 3 epochs -- 10x/3x v1): genuinely more policy movement during training (KL an order of magnitude larger, in-training val reward up to 0.58), but landed on the **exact same failure-type breakdown as v1** on the held-out eval -- none of that extra movement transferred.

Two independently-tuned configs converging on an identical result is stronger evidence of a real plateau than either alone. The online generate-grade-update loop itself works correctly end to end (the actual capability gap this stage set out to demonstrate) -- this particular recipe (500 GSM8K prompts, exact-match + format reward, continuing an already-strong DPO policy) just isn't moving this eval further within the LR/epoch range tried. Concluded the stage here rather than run a v3. Full detail in `DECISIONS.md` Decisions 026-028.

## Next Step
GRPO stage concluded. Move to a different next stage: serving/inference comparison (vLLM/TensorRT-LLM), or synthetic/self-distilled data generation.

Targeted failure-mode SFT data (hand-curated, based on the Qwen failure patterns below) is still a separate, later addition on top of the general GSM8K data.
