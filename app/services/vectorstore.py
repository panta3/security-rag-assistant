import chromadb

from app.core.config import settings


class VectorStore:
    def __init__(self):
        # PersistentClient writes to disk at settings.vector_db_path, so
        # ingested data survives across server restarts.
        self._client = chromadb.PersistentClient(path=settings.vector_db_path)
        self._collection = self._client.get_or_create_collection("security_docs")

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        self._collection.add(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    def query(self, embedding: list[float], top_k: int = 5) -> list[dict]:
        results = self._collection.query(query_embeddings=[embedding], n_results=top_k)

        # Chroma returns parallel lists — one outer entry per query
        # embedding we passed in, so [0] unwraps to "the one query we sent."
        hits = []
        for doc, meta, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            hits.append({"text": doc, "metadata": meta, "distance": distance})
        return hits


# Module-level singleton, same reasoning as embedding_model — one client
# connection reused across requests, not reopened per request.
vector_store = VectorStore()
