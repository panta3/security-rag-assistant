FROM python:3.12-slim

WORKDIR /app

# llama-cpp-python has no guaranteed prebuilt wheel for every platform —
# it falls back to compiling its C++ core from source, which needs an
# actual compiler. python:3.12-slim doesn't ship one.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# --extra-index-url pulls the CPU-only torch build (no bundled CUDA) —
# Cloud Run's standard tier has no GPU, and the default PyPI torch wheel
# would otherwise add multiple unused GBs to the image and cold start.
# torch/transformers are still needed for the embedding model — only LLM
# generation moved to llama.cpp.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Bake the model weights into the image at build time, before COPY-ing
# our own source — this step doesn't need our code, so editing app/
# later won't bust Docker's cache on this download. Cloud Run instances
# are ephemeral (nothing downloaded at runtime survives to the next cold
# start), so without this, every fresh instance would re-download on its
# first request. The GGUF quantized model (~2GB) replaces the old
# full-precision transformers load (~6GB, and ~0.26 tok/s on CPU —
# unusably slow; llama.cpp gets this to ~10 tok/s).
RUN python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='Qwen/Qwen2.5-3B-Instruct-GGUF', filename='qwen2.5-3b-instruct-q4_k_m.gguf'); \
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .
ENV PYTHONPATH=/app

# Pre-ingest the (fixed, small) NIST CSF corpus into Chroma at build
# time too, for the same reason — no persistent volume across instances,
# and the corpus never changes, so there's nothing to gain from
# re-ingesting on every cold start. This one *does* need our source
# (app/services/*) and the PDF, so it stays after COPY.
RUN python scripts/build_index.py

# Cloud Run injects $PORT — don't hardcode 8000 in prod.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
