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


# Empirically, a genuinely relevant hit against this corpus scores ~0.5-0.65
# distance; a genuinely irrelevant query (e.g. asking about something the
# doc never mentions) scores ~1.9+. Prompt instructions alone don't
# reliably stop a small model from answering off its own general
# knowledge instead of admitting the context doesn't cover it (confirmed:
# asked "what is the capital of France" against a security-doc corpus and
# it happily answered "Paris") — so retrieval relevance is enforced
# structurally here rather than trusted to the model's honesty.
NO_MATCH_DISTANCE_THRESHOLD = 1.0

# Still tells the model to stick to the context — even below the
# threshold, this discourages it from padding a real answer with
# ungrounded elaboration.
PROMPT_TEMPLATE = """Answer the question using ONLY the context below. If the \
context doesn't contain enough information to answer, say so explicitly \
instead of guessing.

Context:
{context}

Question: {question}

Answer:"""


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

    citations = [
        Citation(
            source_doc=h["metadata"]["source_doc"],
            page=h["metadata"]["page"],
            excerpt=h["text"][:200],
        )
        for h in hits
    ]

    return RagAnswer(answer=answer_text, citations=citations)
