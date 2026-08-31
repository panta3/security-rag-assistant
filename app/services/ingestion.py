from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Chunk:
    text: str
    source_doc: str
    page: int


def load_pdf(path: str) -> list[tuple[int, str]]:
    """Returns (page_number, page_text) pairs — keeping the page number
    per-page (not per-document) is what makes citations possible later."""
    reader = PdfReader(path)
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Naive fixed-size word chunking with overlap. Good enough for v1 —
    revisit only if retrieval quality turns out to actually be a chunking
    problem, not a retrieval or generation problem."""
    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += step
    return chunks


def ingest_pdf(path: str, chunk_size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Chunks are built per-page rather than across page boundaries — a
    chunk spanning two pages would have an ambiguous page citation, and
    that's the whole point of tracking pages in the first place."""
    doc_name = Path(path).name
    chunks: list[Chunk] = []

    for page_num, page_text in load_pdf(path):
        for text in chunk_text(page_text, chunk_size, overlap):
            if text.strip():
                chunks.append(Chunk(text=text, source_doc=doc_name, page=page_num))

    return chunks
