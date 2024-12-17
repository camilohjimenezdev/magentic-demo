from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import torch


class TransformerEmbedder:
    """Embedder using sentence-transformers models."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedder with a pre-trained model.

        Args:
            model_name: Name of the pre-trained model to use.
                      Defaults to "all-MiniLM-L6-v2"
        """
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for input text(s).

        Args:
            texts: Single text string or list of text strings to embed

        Returns:
            numpy.ndarray: Embeddings matrix of shape (n_texts, embedding_dim)
        """
        return self.model.encode(texts)

    def similarity(
        self, texts1: Union[str, List[str]], texts2: Union[str, List[str]]
    ) -> np.ndarray:
        """Calculate cosine similarity between two sets of texts.

        Args:
            texts1: First text or list of texts
            texts2: Second text or list of texts

        Returns:
            numpy.ndarray: Similarity matrix
        """
        embeddings1 = self.embed(texts1)
        embeddings2 = self.embed(texts2)

        # Convert to torch tensors for efficient similarity computation
        embeddings1 = torch.FloatTensor(embeddings1)
        embeddings2 = torch.FloatTensor(embeddings2)

        # Normalize the embeddings
        embeddings1 = torch.nn.functional.normalize(embeddings1, p=2, dim=1)
        embeddings2 = torch.nn.functional.normalize(embeddings2, p=2, dim=1)

        # Calculate cosine similarity
        similarity_matrix = torch.mm(embeddings1, embeddings2.transpose(0, 1))

        return similarity_matrix.numpy()
