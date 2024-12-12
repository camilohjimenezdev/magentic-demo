import numpy as np
from typing import List

class PlaceholderEmbedder:
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
    
    def embed_text(self, text: str) -> List[float]:
        """Generate placeholder random embedding"""
        return list(np.random.rand(self.embedding_dim))