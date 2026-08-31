from dataclasses import dataclass

from app.services.embeddings import embedding_model
from app.services.vectorstore import vector_store
from app.services.generation import llm


@dataclass
class Citation:
    source_doc: str
    page: int
    excerpt: str


@dataclass
class RagAnswer:
    answer: str
    citations: list[Citation]


# Grounding is defended in two layers, because a single distance
# threshold can't do the whole job — measured across a 28-question eval
# set, genuinely answerable questions scored up to 1.385, while questions
# that reuse real domain vocabulary but ask for a fact the doc never
# states (e.g. "minimum number of firewalls required") scored as low as
# 0.956, well inside the answerable range. No threshold cleanly separates
# those two cases, because both retrieve real, topically relevant content.
#
# Layer 1 (here): block only queries with no relevant content at all —
# threshold set above the highest observed answerable distance (1.385)
# but below the lowest observed genuinely-unrelated distance (1.61).
# Layer 2 (the prompt + _is_decline below): catches the harder case —
# relevant-looking context that doesn't actually contain the answer —
# by requiring the model to say so explicitly rather than guess.
NO_MATCH_DISTANCE_THRESHOLD = 1.5

PROMPT_TEMPLATE = """Answer the question using ONLY the context below. If the \
context doesn't contain enough information to answer, say so explicitly \
instead of guessing.

Context:
{context}

Question: {question}

Answer:"""

_DECLINE_PHRASES = [
    "don't have enough information",
    "do not have enough information",
    "does not contain",
    "doesn't contain",
    "does not specify",
    "doesn't specify",
    "not specify",
    "no specific mention",
    "no mention",
    "not mentioned",
    "cannot determine",
    "can't determine",
    "no information",
]


def _is_decline(answer: str) -> bool:
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in _DECLINE_PHRASES)


def answer_query(query: str, top_k: int = 5) -> RagAnswer:
    q_embedding = embedding_model.embed([query])[0]
    hits = vector_store.query(q_embedding, top_k=top_k)

    if not hits or hits[0]["distance"] > NO_MATCH_DISTANCE_THRESHOLD:
        return RagAnswer(
            answer="I don't have enough information in the ingested documents to answer this.",
            citations=[],
        )

    context = "\n\n".join(
        f"[{i + 1}] (source: {h['metadata']['source_doc']}, "
        f"page {h['metadata']['page']})\n{h['text']}"
        for i, h in enumerate(hits)
    )

    prompt = PROMPT_TEMPLATE.format(context=context, question=query)
    answer_text = llm.generate(prompt)

    # If the model declined in its own words, it didn't actually rely on
    # these chunks — citing them anyway would misrepresent an "I don't
    # know" as a grounded answer.
    if _is_decline(answer_text):
        return RagAnswer(answer=answer_text, citations=[])

    citations = [
        Citation(
            source_doc=h["metadata"]["source_doc"],
            page=h["metadata"]["page"],
            excerpt=h["text"][:200],
        )
        for h in hits
    ]

    return RagAnswer(answer=answer_text, citations=citations)
