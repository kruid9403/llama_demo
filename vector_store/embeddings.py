import os

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/e5-small-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

class LocalEmbedder:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)
        self.dimension = int(self.model.get_sentence_embedding_dimension())
        if self.dimension != EMBEDDING_DIM:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"model '{EMBEDDING_MODEL_NAME}' outputs {self.dimension}, "
                f"but EMBEDDING_DIM={EMBEDDING_DIM}. "
                "Use a 384-d model (for example, intfloat/e5-small-v2) or "
                "change EMBEDDING_DIM and reinitialize the vector schema."
            )

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
