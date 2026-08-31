import uuid

from fastapi import APIRouter
from pydantic import BaseModel

from app.services.ingestion import ingest_pdf
from app.services.embeddings import embedding_model
from app.services.vectorstore import vector_store
from app.services.rag import answer_query

router = APIRouter()


class QueryRequest(BaseModel):
    question: str


class IngestRequest(BaseModel):
    path: str  # path to a PDF under data/


@router.post("/ingest")
def ingest(req: IngestRequest):
    chunks = ingest_pdf(req.path)
    if not chunks:
        return {"chunks_ingested": 0}

    embeddings = embedding_model.embed([c.text for c in chunks])

    vector_store.add(
        ids=[str(uuid.uuid4()) for _ in chunks],
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
