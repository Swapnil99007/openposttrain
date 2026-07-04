# GSM8K Experiment Report

## Run Summary

|       timestamp | run_name                                        | model_name                 |   limit |   accuracy | prompt_path          | config_path                              |
|----------------:|:------------------------------------------------|:---------------------------|--------:|-----------:|:---------------------|:-----------------------------------------|
| 20260704_012224 | gsm8k_qwen2_5_1_5b_instruct_baseline            | Qwen/Qwen2.5-1.5B-Instruct |     100 |       0.43 | prompts/gsm8k_v1.txt | configs/eval_gsm8k_qwen2_5_1_5b.yaml     |
| 20260704_013819 | gsm8k_qwen2_5_1_5b_instruct_baseline_512_tokens | Qwen/Qwen2.5-1.5B-Instruct |     100 |       0.7  | prompts/gsm8k_v1.txt | configs/eval_gsm8k_qwen2_5_1_5b_512.yaml |

## Failure Type Summary

|       timestamp | run_name                                        | prompt_path          |   correct |   no_numeric_answer |   format_violation |   wrong_numeric_answer |
|----------------:|:------------------------------------------------|:---------------------|----------:|--------------------:|-------------------:|-----------------------:|
| 20260704_012224 | gsm8k_qwen2_5_1_5b_instruct_baseline            | prompts/gsm8k_v1.txt |        43 |                   0 |                 54 |                      3 |
| 20260704_013819 | gsm8k_qwen2_5_1_5b_instruct_baseline_512_tokens | prompts/gsm8k_v1.txt |        70 |                   0 |                 18 |                     12 |

## Notes

- `tiny-gpt2` is only a smoke-test model.
- SmolLM2-135M is instruction-tuned but still weak on GSM8K.
- Prompt v1 currently has fewer formatting violations than prompt v2 strict.