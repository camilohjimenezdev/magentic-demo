import numpy as np


class PlaceholderEmbedder:
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        np.random.seed(42)  # For consistent test results

    def embed_text(self, text: str) -> np.ndarray:
        # Generate a deterministic embedding based on the text
        np.random.seed(hash(text) % 2**32)
        embedding = np.random.rand(self.embedding_dim)
        # Normalize the embedding
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
