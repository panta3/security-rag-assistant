# Security RAG Assistant

RAG pipeline over security/compliance documents (policies, CIS
benchmarks, framework docs) — answers questions with citations back to
the source doc, and measures its own hallucination rate instead of just
trusting the output. Runs locally today; GCP Cloud Run is the deploy
target (see TODO.md), not yet live.

## Why security docs, not generic PDFs
Ties this project to the Cloud Security pillar instead of being a
standalone "RAG chatbot" — one of the most generic project types on
resumes right now. The security-domain framing plus the eval harness is
what makes it defensible in an interview.

## Architecture
```
PDF ingestion -> chunking -> embeddings -> vector DB
                                              |
User query -> retrieval (top-k) -> LLM generation (with citations) -> answer
                                              |
                                    eval harness (retrieval precision,
                                    citation accuracy, hallucination rate)
```

## Stack
- FastAPI (serving) + a single self-contained HTML/CSS/JS frontend at `/`
  (no framework — the whole point is a URL to click, not another curl call)
- PyTorch + sentence-transformers (embeddings), GPU-accelerated
- Chroma (vector DB, local persistent — zero setup)
- Qwen2.5-3B-Instruct (generation), running locally via transformers —
  no external API key needed
- GCP Cloud Run (deploy target)
- Docker

## Core features (MVP scope)
- [x] Ingest real security/compliance PDFs (NIST CSF 2.0 as the test corpus)
- [x] Chunk + embed + store in vector DB
- [x] `/query` endpoint: retrieve top-k chunks, generate an answer with
      inline citations back to source doc + page
- [x] Hallucination guard, two layers: a retrieval-distance threshold
      blocks queries with no relevant content at all, and the model is
      required to say so explicitly when retrieval finds related-but-
      insufficient context — measured, not just "the model was honest
      that one time"
- [x] Eval harness: 28 hand-built questions against the real ingested
      corpus — **66.7% retrieval precision, 66.7% citation accuracy,
      66.7% keyword recall, 0% hallucination rate** (reproducible across
      repeated runs)

## Explicitly out of scope for v1
- Multi-turn conversation / chat history
- User auth
- Fine-tuning anything — this is retrieval quality, not model training

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# data/*.pdf is gitignored (test corpus, not source code) — grab a real
# public security doc, e.g. NIST CSF 2.0:
curl -sSL -o data/nist_csf_2.0.pdf https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf

uvicorn app.main:app --reload
# then: POST /ingest {"path": "data/nist_csf_2.0.pdf"}, then POST /query {"question": "..."}
# or: POST /eval {} to run the full eval harness against data/eval_set.json
```

Requires a CUDA GPU for reasonable generation speed (falls back to CPU
automatically via `device_map="auto"`, but will be slow). First run
downloads the embedding model (~90MB) and Qwen2.5-3B-Instruct (~6GB)
from Hugging Face.

## Status
Ingestion, retrieval, generation, the eval harness (real measured
numbers, see above), and a minimal frontend are all working end-to-end
locally. Not yet deployed — Cloud Run is next. See `TODO.md`.
