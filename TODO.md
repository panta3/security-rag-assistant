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
- [ ] `app/services/eval.py` — build a small labeled Q&A set by hand (20-30 questions with known correct source chunks), measure retrieval precision + citation accuracy
- [ ] Formal hallucination rate metric across the eval set (the threshold guard above is a structural fix, not yet a measured number)

## December
- [ ] Dockerize, deploy to Cloud Run
- [ ] README screenshots/demo + eval results
- [ ] Resume bullet once deployed and eval numbers exist

## Notes
- Don't skip the eval harness — "I built a RAG app" is generic, "I measured X% retrieval precision and Y% citation accuracy" is not.
