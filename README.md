# Security RAG Assistant

RAG pipeline over security/compliance documents (policies, CIS
benchmarks, framework docs) — answers questions with citations back to
the source doc, and tracks its own hallucination rate instead of just
trusting the output. Deployed on GCP Cloud Run.

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
- FastAPI (serving)
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
- [x] Hallucination guard: refuses to answer (no LLM call at all) when
      retrieval doesn't find anything relevant, rather than trusting the
      model to admit it doesn't know
- [ ] Eval harness: a small labeled question set, measure retrieval
      precision and whether answers cite real (vs. hallucinated) sources

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
```

Requires a CUDA GPU for reasonable generation speed (falls back to CPU
automatically via `device_map="auto"`, but will be slow). First run
downloads the embedding model (~90MB) and Qwen2.5-3B-Instruct (~6GB)
from Hugging Face.

## Status
Ingestion, retrieval, and generation all working end-to-end, validated
against a real NIST CSF 2.0 PDF. Eval harness (formal precision/citation/
hallucination-rate metrics) not yet built. See `TODO.md`.
