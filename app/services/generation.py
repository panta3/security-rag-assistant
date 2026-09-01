import os

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# Q4_K_M: 4-bit quantization, the standard "good default" balance of size,
# speed, and quality for llama.cpp — plain fp16/fp32 transformers on CPU
# measured at ~0.26 tok/s for this model (a 512-token reply would take
# 30+ minutes), which is what this class of quantized CPU inference exists
# to fix.
_MODEL_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
_MODEL_FILE = "qwen2.5-3b-instruct-q4_k_m.gguf"


class LocalLLM:
    def __init__(self):
        # Downloads once (cached by huggingface_hub) — baked into the
        # Docker image at build time in production, same reasoning as the
        # old transformers-based load: Cloud Run instances are ephemeral,
        # so nothing downloaded at runtime survives to the next cold start.
        model_path = hf_hub_download(repo_id=_MODEL_REPO, filename=_MODEL_FILE)
        self._llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=os.cpu_count(),
            verbose=False,
        )

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        # create_chat_completion applies the model's own chat template
        # (read from the GGUF's embedded metadata) — no separate tokenizer
        # or chat-template handling needed on our side.
        result = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=0.0,
        )
        return result["choices"][0]["message"]["content"]


llm = LocalLLM()
