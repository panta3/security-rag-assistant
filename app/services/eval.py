"""Eval harness — this is what makes the project defensible, don't skip it.

For each answerable question: does top-k retrieval surface an expected
page (retrieval precision)? Does the final answer cite an expected page
(citation accuracy)? Does the answer text contain an expected keyword, as
a cheap proxy for "did it actually answer correctly" (keyword recall)?

For each deliberately unanswerable question (real questions, but nothing
in the corpus supports an answer): did the system correctly decline
instead of answering confidently off its own knowledge? That's the
hallucination rate — the same failure mode the "capital of France" bug
caught, now measured as a number instead of one anecdote.
"""

import json

from app.services.embeddings import embedding_model
from app.services.vectorstore import vector_store
from app.services.rag import answer_query


def evaluate(eval_set_path: str, top_k: int = 5) -> dict:
    with open(eval_set_path) as f:
        eval_set = json.load(f)

    answerable = [q for q in eval_set if q["answerable"]]
    unanswerable = [q for q in eval_set if not q["answerable"]]

    retrieval_hits = 0
    citation_hits = 0
    keyword_hits = 0

    for item in answerable:
        q_embedding = embedding_model.embed([item["question"]])[0]
        hits = vector_store.query(q_embedding, top_k=top_k)
        retrieved_pages = {h["metadata"]["page"] for h in hits}
        if retrieved_pages & set(item["expected_pages"]):
            retrieval_hits += 1

        result = answer_query(item["question"], top_k=top_k)
        cited_pages = {c.page for c in result.citations}
        if cited_pages & set(item["expected_pages"]):
            citation_hits += 1

        answer_lower = result.answer.lower()
        if any(kw.lower() in answer_lower for kw in item["expected_keywords"]):
            keyword_hits += 1

    # answer_query() returns citations=[] on both decline paths (distance
    # threshold, or the model declining in its own words) — see rag.py —
    # so an empty citation list reliably means "declined," non-empty
    # reliably means "answered." A hallucination is a genuinely
    # unanswerable question the system answered anyway.
    hallucinated = 0
    for item in unanswerable:
        result = answer_query(item["question"], top_k=top_k)
        if result.citations:
            hallucinated += 1

    n_answerable = len(answerable)
    n_unanswerable = len(unanswerable)

    return {
        "n_answerable": n_answerable,
        "n_unanswerable": n_unanswerable,
        "retrieval_precision": retrieval_hits / n_answerable if n_answerable else None,
        "citation_accuracy": citation_hits / n_answerable if n_answerable else None,
        "keyword_recall": keyword_hits / n_answerable if n_answerable else None,
        "hallucination_rate": hallucinated / n_unanswerable if n_unanswerable else None,
    }
