# Security RAG Assistant

**Live: https://security-rag-assistant-x3hib67cua-uc.a.run.app**

RAG pipeline over security/compliance documents (policies, CIS
benchmarks, framework docs) — answers questions with citations back to
the source doc, and measures its own hallucination rate instead of just
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
- FastAPI (serving) + a single self-contained HTML/CSS/JS frontend at `/`
  (no framework — the whole point is a URL to click, not another curl call)
- PyTorch + sentence-transformers (embeddings) — GPU-accelerated locally,
  CPU in production (the embedding model is small; it was never the
  bottleneck)
- Chroma (vector DB, local persistent — zero setup)
- Qwen2.5-3B-Instruct (generation) — no external API key needed.
  Quantized (GGUF, Q4_K_M) via `llama-cpp-python` in production for
  usable CPU inference speed (~10 tok/s vs. ~0.26 tok/s for the naive
  full-precision path — see TODO.md); full-precision via `transformers`
  for local GPU dev
- GCP Cloud Run + Docker (deployed, CPU-only — see TODO.md for why GPU
  Cloud Run wasn't the fix, and what actually was)

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

# data/nist_csf_2.0.pdf is committed (public NIST document, no license
# issue) — it's the corpus the Docker build bakes into the image.

uvicorn app.main:app --reload
# then: POST /ingest {"path": "data/nist_csf_2.0.pdf"}, then POST /query {"question": "..."}
# or: POST /eval {} to run the full eval harness against data/eval_set.json
```

Local dev uses full-precision `transformers` and prefers a CUDA GPU for
reasonable generation speed (falls back to CPU automatically via
`device_map="auto"`, but will be slow — this is exactly the problem the
deployed version's GGUF quantization solves). First run downloads the
embedding model (~90MB) and Qwen2.5-3B-Instruct (~6GB) from Hugging Face.

To build/run the same container the deployed version uses:
```bash
docker build -t security-rag-assistant .
docker run -p 8080:8080 security-rag-assistant
```

## Status
Fully deployed and live (see the link at the top). Ingestion, retrieval,
generation, the eval harness (real measured numbers, see above), and the
frontend are all working end-to-end in production, not just locally.
See `TODO.md` for what's left (README demo screenshots) and for the real
bugs found getting this deployed — worth reading if you want the honest
version, not just the checklist.
