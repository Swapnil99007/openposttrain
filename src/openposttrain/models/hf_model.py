from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


DTYPE_NAMES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class HFModel:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        adapter_path: str | None = None,
        dtype: str | None = None,
    ):
        self.model_name = model_name
        self.adapter_path = adapter_path

        # If training patched in a chat template (e.g. via chat_template_path,
        # needed for base models with no template of their own), that patch
        # is saved into the adapter's output directory, not back into the
        # original base model repo. Load the tokenizer from there when an
        # adapter is present so eval sees the same template training used.
        tokenizer_source = adapter_path if adapter_path else model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if dtype:
            resolved_dtype = DTYPE_NAMES[dtype]
        else:
            resolved_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=resolved_dtype,
            device_map=device,
        )

        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float | None = None,
        no_repeat_ngram_size: int | None = None,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = prompt

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        do_sample = temperature > 0.0

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        # Base (non-chat-tuned) models have no natural turn-boundary signal
        # and can degenerate into repeating a single token under greedy
        # decoding. These are opt-in (None = unchanged behavior) so existing
        # instruct-model eval configs aren't retroactively affected.
        if repetition_penalty is not None:
            generation_kwargs["repetition_penalty"] = repetition_penalty

        if no_repeat_ngram_size is not None:
            generation_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response.strip()
