# embeddings.py
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"

class LocalEmbedder:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def embed_query(self, text: str):
        return self.model.encode(
            f"query: {text}",
            normalize_embeddings=True,
        ).tolist()

    def embed_passages(self, texts):
        return self.model.encode(
            [f"passage: {t}" for t in texts],
            normalize_embeddings=True,
        ).tolist()
