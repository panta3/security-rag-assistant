import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.core.config import settings


class LocalLLM:
    def __init__(self):
        # Loaded once at import time, same reasoning as EmbeddingModel —
        # this is expensive (model weights onto the GPU), never per-request.
        self._tokenizer = AutoTokenizer.from_pretrained(settings.llm_model)
        self._model = AutoModelForCausalLM.from_pretrained(
            settings.llm_model,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        messages = [{"role": "user", "content": prompt}]
        # return_dict=True gives back input_ids + attention_mask together —
        # generate() needs both (a raw input_ids tensor alone silently
        # omits the attention mask, which is wrong even when it happens
        # not to crash).
        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self._model.device)

        output = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        # Slice off the echoed prompt tokens — generate() returns the full
        # sequence (prompt + completion), we only want the new part.
        new_tokens = output[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)


llm = LocalLLM()
