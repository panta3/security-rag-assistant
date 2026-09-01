"""
Runs the same ingestion logic as POST /ingest, but as a one-shot script
callable during `docker build` — the corpus (data/nist_csf_2.0.pdf) is
fixed and doesn't change, so there's no reason to make every fresh Cloud
Run instance re-run ingestion at cold start (there's also no persistent
volume for it to write into across instances). This bakes the resulting
Chroma index directly into the image.
"""

from collections import defaultdict

from app.services.embeddings import embedding_model
from app.services.ingestion import ingest_pdf
from app.services.vectorstore import vector_store


def build(pdf_path: str = "data/nist_csf_2.0.pdf") -> None:
    chunks = ingest_pdf(pdf_path)
    if not chunks:
        raise RuntimeError(f"No chunks extracted from {pdf_path} — check the PDF is present.")

    embeddings = embedding_model.embed([c.text for c in chunks])

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

    print(f"Ingested {len(chunks)} chunks from {pdf_path}")


if __name__ == "__main__":
    build()
