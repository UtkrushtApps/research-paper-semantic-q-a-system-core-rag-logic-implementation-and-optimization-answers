# rag_pipeline/embedding.py
from sentence_transformers import SentenceTransformer
from threading import Lock

class EmbeddingModel:
    _model = None
    _lock = Lock()

    @classmethod
    def get_model(cls, model_name):
        with cls._lock:
            if cls._model is None:
                cls._model = SentenceTransformer(model_name)
            return cls._model

def embed_texts(texts, model_name):
    """Compute embeddings for a list of texts using specified model."""
    model = EmbeddingModel.get_model(model_name)
    return model.encode(texts, show_progress_bar=False, batch_size=32, convert_to_numpy=True)
