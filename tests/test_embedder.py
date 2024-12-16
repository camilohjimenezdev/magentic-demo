"""
Test suite for the PlaceholderEmbedder class.

This module contains unit tests to verify the functionality of the PlaceholderEmbedder,
which is a simple text embedding generator used for testing or placeholder purposes.
"""

import numpy as np
from app.embedding.embedder import PlaceholderEmbedder


def test_embed_text():
    """
    Test basic text embedding functionality.

    Verifies that:
    1. The embedder returns a numpy array
    2. The embedding has the correct dimension (384)
    3. The embedding is not a zero vector
    """
    embedder = PlaceholderEmbedder(embedding_dim=384)
    text = "This is a test sentence"

    embedding = embedder.embed_text(text)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (384,)
    assert not np.all(embedding == 0)


def test_consistent_embedding():
    """
    Test embedding consistency.

    Verifies that the same input text produces identical embeddings
    when processed multiple times by the same embedder instance.
    """
    embedder = PlaceholderEmbedder(embedding_dim=384)
    text = "Test sentence"

    embedding1 = embedder.embed_text(text)
    embedding2 = embedder.embed_text(text)

    assert np.array_equal(embedding1, embedding2)


def test_different_dimensions():
    """
    Test embedder with different dimensions.

    Verifies that the embedder can be initialized with different
    embedding dimensions and produces vectors of the specified size.
    """
    embedder = PlaceholderEmbedder(embedding_dim=128)
    text = "Test sentence"

    embedding = embedder.embed_text(text)
    assert embedding.shape == (128,)
