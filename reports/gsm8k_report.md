# GSM8K Experiment Report

## Run Summary

|       timestamp | run_name                                        | model_name                          |   limit |   accuracy | prompt_path                 | config_path                              |
|----------------:|:------------------------------------------------|:------------------------------------|--------:|-----------:|:----------------------------|:-----------------------------------------|
|                 |                                                 | sshleifer/tiny-gpt2                 |       3 |       0    |                             |                                          |
|                 | gsm8k_tiny_gpt2_smoke_test                      | sshleifer/tiny-gpt2                 |       3 |       0    |                             | configs/eval_gsm8k_tiny.yaml             |
|                 | gsm8k_tiny_gpt2_smoke_test                      | sshleifer/tiny-gpt2                 |       3 |       0    |                             | configs/eval_gsm8k_tiny.yaml             |
| 20260703_150712 | gsm8k_tiny_gpt2_smoke_test                      | sshleifer/tiny-gpt2                 |       3 |       0    |                             | configs/eval_gsm8k_tiny.yaml             |
| 20260703_151220 | gsm8k_tiny_gpt2_smoke_test                      | sshleifer/tiny-gpt2                 |       3 |       0    |                             | configs/eval_gsm8k_tiny.yaml             |
| 20260703_151856 | gsm8k_tiny_gpt2_smoke_test                      | sshleifer/tiny-gpt2                 |       3 |       0    |                             | configs/eval_gsm8k_tiny.yaml             |
| 20260703_152505 | gsm8k_smollm2_135m_instruct_local               | HuggingFaceTB/SmolLM2-135M-Instruct |      10 |       0    |                             | configs/eval_gsm8k_smollm2_135m.yaml     |
| 20260703_153553 | gsm8k_smollm2_135m_instruct_strict_prompt       | HuggingFaceTB/SmolLM2-135M-Instruct |      10 |       0    | prompts/gsm8k_v2_strict.txt | configs/eval_gsm8k_smollm2_135m.yaml     |
| 20260703_165414 | gsm8k_smollm2_135m_instruct_prompt_v1           | HuggingFaceTB/SmolLM2-135M-Instruct |      10 |       0    | prompts/gsm8k_v1.txt        | configs/eval_gsm8k_smollm2_135m_v1.yaml  |
| 20260704_012224 | gsm8k_qwen2_5_1_5b_instruct_baseline            | Qwen/Qwen2.5-1.5B-Instruct          |     100 |       0.43 | prompts/gsm8k_v1.txt        | configs/eval_gsm8k_qwen2_5_1_5b.yaml     |
| 20260704_013819 | gsm8k_qwen2_5_1_5b_instruct_baseline_512_tokens | Qwen/Qwen2.5-1.5B-Instruct          |     100 |       0.7  | prompts/gsm8k_v1.txt        | configs/eval_gsm8k_qwen2_5_1_5b_512.yaml |

## Failure Type Summary

|       timestamp | run_name                                        | prompt_path                 |   correct |   no_numeric_answer |   format_violation |   wrong_numeric_answer |
|----------------:|:------------------------------------------------|:----------------------------|----------:|--------------------:|-------------------:|-----------------------:|
| 20260703_151220 | gsm8k_tiny_gpt2_smoke_test                      |                             |         0 |                   3 |                  0 |                      0 |
| 20260703_151856 | gsm8k_tiny_gpt2_smoke_test                      |                             |         0 |                   3 |                  0 |                      0 |
| 20260703_152505 | gsm8k_smollm2_135m_instruct_local               |                             |         0 |                   0 |                  3 |                      7 |
| 20260703_153553 | gsm8k_smollm2_135m_instruct_strict_prompt       | prompts/gsm8k_v2_strict.txt |         0 |                   0 |                  8 |                      2 |
| 20260703_165414 | gsm8k_smollm2_135m_instruct_prompt_v1           | prompts/gsm8k_v1.txt        |         0 |                   0 |                  1 |                      9 |
| 20260704_012224 | gsm8k_qwen2_5_1_5b_instruct_baseline            | prompts/gsm8k_v1.txt        |        43 |                   0 |                 54 |                      3 |
| 20260704_013819 | gsm8k_qwen2_5_1_5b_instruct_baseline_512_tokens | prompts/gsm8k_v1.txt        |        70 |                   0 |                 18 |                     12 |

## Notes

- `tiny-gpt2` is only a smoke-test model.
- SmolLM2-135M is instruction-tuned but still weak on GSM8K.
- Prompt v1 currently has fewer formatting violations than prompt v2 strict.