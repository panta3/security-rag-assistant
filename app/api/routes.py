from collections import defaultdict

from fastapi import APIRouter
from pydantic import BaseModel

from app.services.ingestion import ingest_pdf
from app.services.embeddings import embedding_model
from app.services.vectorstore import vector_store
from app.services.rag import answer_query
from app.services.eval import evaluate

router = APIRouter()


class QueryRequest(BaseModel):
    question: str


class IngestRequest(BaseModel):
    path: str  # path to a PDF under data/


class EvalRequest(BaseModel):
    eval_set_path: str = "data/eval_set.json"


@router.post("/ingest")
def ingest(req: IngestRequest):
    chunks = ingest_pdf(req.path)
    if not chunks:
        return {"chunks_ingested": 0}

    embeddings = embedding_model.embed([c.text for c in chunks])

    # Deterministic per doc+page+index-within-page, not a random UUID —
    # re-ingesting the same PDF upserts existing chunks instead of
    # duplicating them (chroma_data persists across restarts).
    page_counts: dict[int, int] = defaultdict(int)
    ids = []
    for c in chunks:
        idx = page_counts[c.page]
        page_counts[c.page] += 1
        ids.append(f"{c.source_doc}#{c.page}#{idx}")

    vector_store.add(
        ids=ids,
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[{"source_doc": c.source_doc, "page": c.page} for c in chunks],
    )

    return {"chunks_ingested": len(chunks)}


@router.post("/query")
def query(req: QueryRequest):
    result = answer_query(req.question)
    return {
        "answer": result.answer,
        "citations": [
            {
                "source_doc": c.source_doc,
                "page": c.page,
                "excerpt": c.excerpt,
            }
            for c in result.citations
        ],
    }


@router.post("/eval")
def eval_endpoint(req: EvalRequest):
    return evaluate(req.eval_set_path)
