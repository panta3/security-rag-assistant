from sentence_transformers import SentenceTransformer

from app.core.config import settings


class EmbeddingModel:
    def __init__(self):
        # Model weights load once here, not per-request — SentenceTransformer
        # auto-detects and uses the GPU if one's available.
        self._model = SentenceTransformer(settings.embedding_model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, convert_to_numpy=True).tolist()


# Module-level singleton — routes import this directly instead of
# constructing (and reloading) a model per request.
embedding_model = EmbeddingModel()
