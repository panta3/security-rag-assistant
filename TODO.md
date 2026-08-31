# TODO — Security RAG Assistant

## October
- [x] Real security/compliance PDF in `data/`: NIST Cybersecurity Framework 2.0 (public, no gating — CIS benchmarks are gated behind email signup)
- [x] `app/services/ingestion.py` — page-aware PDF parsing + chunking (chunks never span page boundaries, so citations stay unambiguous)
- [x] `app/services/embeddings.py` — sentence-transformers (all-MiniLM-L6-v2), GPU-accelerated (RTX 3070 Ti)
- [x] `app/services/vectorstore.py` — Chroma, persistent
- [x] `/ingest` endpoint — tested against the real NIST PDF, 33 chunks, retrieval quality confirmed good

## November
- [x] `app/services/generation.py` — local Qwen2.5-3B-Instruct on GPU (no API key needed)
- [x] `app/services/rag.py` — retrieval + generation + citation formatting
- [x] `/query` endpoint — tested end-to-end, correct grounded answers with citations
- [x] Hallucination guard: discovered prompt instructions alone don't reliably stop the model answering off its own knowledge (asked "capital of France" against the security corpus, it said "Paris"). Fixed structurally — distance threshold on retrieval (empirically ~0.5-0.65 for relevant hits vs ~1.9+ for irrelevant ones) short-circuits generation entirely when nothing relevant was retrieved, rather than trusting the model's honesty.
- [x] `app/services/eval.py` + `data/eval_set.json` — 28 hand-built questions (24 answerable across the real NIST CSF 2.0 doc, 4 deliberately unanswerable) with real page references, not fabricated
- [x] Formal hallucination rate metric: **0%** (0/4) on the unanswerable set
- [x] Retrieval precision: **66.7%**, citation accuracy: **66.7%**, keyword recall: **66.7%** — reproducible across repeated runs (greedy decoding, no randomness)

## December
- [x] Minimal frontend (`static/index.html`, served at `/` by FastAPI) — single self-contained page, no framework: question box, example chips, answer + citations rendered live via fetch() against the existing /query endpoint
- [ ] Dockerize, deploy to Cloud Run (frontend + API both come along for free once that's done, since it's served by the same app)
- [ ] README screenshots/demo + eval results
- [ ] Resume bullet once deployed and eval numbers exist

## Notes
- Don't skip the eval harness — "I built a RAG app" is generic, "I measured X% retrieval precision and Y% citation accuracy" is not.
- Building the eval harness surfaced 3 real bugs, none in the core retrieval/generation logic:
  1. **My own eval set had wrong page numbers.** I confused the NIST document's own printed page footer (e.g. "26") with the PDF's physical page index our ingestion pipeline actually cites (there's a 5-page offset from the cover/abstract/TOC/preface). 10 of 24 expected_pages were off by exactly 5. This alone made retrieval_precision look like 29% when it was actually 50%.
  2. **`/ingest` wasn't idempotent** — chunks were keyed by `uuid.uuid4()`, so re-ingesting the same PDF (which I did repeatedly while testing) silently duplicated every chunk instead of overwriting it. Caught when a "clean" ingest showed 198 chunks in the vector store instead of 33. Fixed the same way as the Posture Scanner's DynamoDB bug: deterministic IDs (`source_doc#page#index`) plus Chroma's `upsert()` instead of `add()`.
  3. **The hallucination-rate detection logic itself was too crude.** Checking "citations non-empty" as the hallucination signal was wrong — the model can legitimately retrieve same-topic-but-insufficient context and still correctly decline in its own answer text (e.g. "the CSF does not specify a minimum number of firewalls"). Moved decline-detection into `rag.py` itself (`_is_decline`, checked against a phrase list) so both `answer_query()` and `evaluate()` share one definition of "declined."
- The distance threshold (`NO_MATCH_DISTANCE_THRESHOLD` in rag.py) was originally calibrated off two anecdotal examples and set too low (1.0), which the real eval set proved was blocking legitimately answerable questions (up to 1.385 distance). Recalibrated to 1.5 using the actual measured distribution. Also confirmed empirically: same-domain-but-fabricated questions (e.g. asking for a specific number the CSF never states) score in the *same* distance range as genuinely answerable questions (0.96–0.99) — no distance threshold can separate those two cases, which is exactly why the two-layer defense (threshold + prompt-level self-awareness) matters instead of relying on retrieval distance alone.
- Never run a standalone script against `chroma_data` while the FastAPI server is also running — two processes writing to the same on-disk Chroma SQLite file concurrently produced a corrupted count (198 chunks from what should have been a single clean 33-chunk ingest). Use the running server's own endpoints for everything once it's up.
