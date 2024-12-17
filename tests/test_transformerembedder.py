"""
Test suite for the TransformerEmbedder class.

This module contains unit tests to verify the functionality of the TransformerEmbedder,
which uses sentence-transformers to generate text embeddings.

0. Basic Validation: Do sentences with similar meaning produce embeddings with high cosine similarity?
1. Semantic Similarity: Do sentences with similar meaning produce embeddings with high cosine similarity?
2. Paraphrase Detection: Can the model identify paraphrased sentences?
3. Sentence Contradiction: Can the model distinguish between unrelated or contradictory sentences?
4. Unusual Inputs: How does the model handle unusual inputs?
"""

import pytest
import numpy as np
from app.embedding.TransformerEmbedder import TransformerEmbedder


@pytest.fixture
def embedder():
    return TransformerEmbedder()


@pytest.fixture
def sample_texts():
    return [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium.",
    ]


def test_embed_single_text(embedder):
    """Test embedding a single text string."""
    text = "This is a test sentence"
    embedding = embedder.embed(text)

    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1  # Should be a 1D array for single text
    assert embedding.shape[0] == 384  # Default dimension for all-MiniLM-L6-v2
    assert not np.all(embedding == 0)  # Should not be all zeros


def test_embed_multiple_texts(embedder, sample_texts):
    """Test embedding multiple texts at once."""
    embeddings = embedder.embed(sample_texts)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.ndim == 2
    assert embeddings.shape == (3, 384)  # 3 texts, 384 dimensions each
    assert not np.all(embeddings == 0)


def test_consistent_embedding(embedder):
    """Test that the same text produces consistent embeddings."""
    text = "Test sentence for consistency"
    embedding1 = embedder.embed(text)
    embedding2 = embedder.embed(text)

    np.testing.assert_array_almost_equal(embedding1, embedding2)


def test_similarity_expected_values(embedder):
    """Test that similar texts have higher similarity scores."""
    texts = [
        "The cat is sleeping",
        "A cat is taking a nap",  # Similar to first text
        "The weather is nice today",  # Different topic
    ]

    similarities = embedder.similarity(texts, texts)

    # First two texts should be more similar to each other
    # than to the third text
    assert similarities[0][1] > similarities[0][2]
    assert similarities[1][0] > similarities[2][0]


@pytest.fixture
def similarity_threshold():
    return 0.7


def test_semantic_similarity_threshold(embedder, similarity_threshold):
    """
    1. Test that semantically similar sentences have high cosine similarity scores.
    2. Uses a threshold called similarity_threshold to determine if sentences are semantically similar.
    3. The difference between similarity and dissimilarity is the threshold.
    """
    # Pairs of semantically similar sentences
    similar_pairs = [
        ("I love programming in Python.", "Python programming is enjoyable."),
        ("The cat is sleeping on the couch.", "A cat is taking a nap on the sofa."),
        ("The weather is beautiful today.", "It's such a lovely day outside."),
    ]

    dissimilar_pairs = [
        ("I love programming in Python.", "The cat is sleeping."),
        ("The weather is beautiful today.", "I need to buy groceries."),
        ("The cat is sleeping on the couch.", "I am flying an airplane."),
    ]

    # All of the dissimilar pairs should have a less similarity score than the similar pairs
    # Calculate the similarity score for each pair of similar pairs and save the scores in an array
    similar_scores = []
    for sentence1, sentence2 in similar_pairs:
        similarity = embedder.similarity([sentence1], [sentence2])[0][0]
        similar_scores.append(similarity)

    # Calculate the similarity score for each pair of dissimilar pairs and save the scores in an array
    dissimilar_scores = []
    for sentence1, sentence2 in dissimilar_pairs:
        similarity = embedder.similarity([sentence1], [sentence2])[0][0]
        dissimilar_scores.append(similarity)

    # All of the dissimilar scores should be less than all the similarity scores
    for dissimilar_score in dissimilar_scores:
        for similar_score in similar_scores:
            assert (
                dissimilar_score < similar_score
            ), f"Expected dissimilar score ({dissimilar_score}) to be less than similar score ({similar_score})"


@pytest.fixture
def similarity_threshold_paraphrases():
    return 0.7


def test_paraphrase_detection(embedder, similarity_threshold_paraphrases):
    """Test the model's ability to detect paraphrases using similarity scores."""
    # Test pairs format: (sentence1, sentence2, is_paraphrase)
    test_pairs = [
        # Paraphrase pairs (should have high similarity)
        (
            "How do I improve my programming skills?",
            "What's the best way to become a better programmer?",
            True,
        ),
        ("The movie was absolutely fantastic!", "I really enjoyed that film.", True),
        (
            "Can you help me with this problem?",
            "Would you mind assisting me with this issue?",
            False,
        ),
        # Non-paraphrase pairs (should have lower similarity)
        (
            "How do I improve my programming skills?",
            "What's the weather like today?",
            False,
        ),
        (
            "The movie was absolutely fantastic!",
            "I need to buy groceries later.",
            False,
        ),
        (
            "Python is my favorite programming language.",
            "The cat is sleeping on the windowsill.",
            False,
        ),
    ]

    for sentence1, sentence2, expected_paraphrase in test_pairs:
        similarity = embedder.similarity([sentence1], [sentence2])[0][0]
        is_paraphrase = similarity > similarity_threshold_paraphrases

        assert is_paraphrase == expected_paraphrase, (
            f"Paraphrase detection failed for:\n"
            f"Sentence 1: {sentence1}\n"
            f"Sentence 2: {sentence2}\n"
            f"Similarity score: {similarity:.3f}\n"
            f"Expected paraphrase: {expected_paraphrase}\n"
            f"Detected paraphrase: {is_paraphrase}"
        )


@pytest.fixture
def similarity_threshold_contradiction():
    return 0.2


def test_contradiction_detection(embedder, similarity_threshold_contradiction):
    """
    Test the model's ability to detect contradictory statements through low similarity scores.
      * Direct negations
      * Opposite weather conditions
      * Opposing preferences
      * Contrary opinions
      * Mutually exclusive states
    """
    # Test pairs format: (sentence1, sentence2, expected_max_similarity)
    contradiction_pairs = [
        (
            "The cat is sleeping on the couch.",
            "There are no cats in the room.",
            similarity_threshold_contradiction,
        ),
        (
            "The weather is sunny today.",
            "It's raining heavily right now.",
            similarity_threshold_contradiction,
        ),
        (
            "I love spicy food.",
            "I hate all spicy dishes.",
            similarity_threshold_contradiction,
        ),
        (
            "The movie was excellent.",
            "The movie was terrible.",
            similarity_threshold_contradiction,
        ),
        (
            "The door is open.",
            "The door is closed.",
            similarity_threshold_contradiction,
        ),
    ]

    for sentence1, sentence2, max_expected_similarity in contradiction_pairs:
        similarity = embedder.similarity([sentence1], [sentence2])[0][0]

        assert similarity < max_expected_similarity, (
            f"Expected contradictory sentences to have low similarity (<{max_expected_similarity}), but got {similarity:.3f}\n"
            f"Sentence 1: {sentence1}\n"
            f"Sentence 2: {sentence2}\n"
            f"Similarity score: {similarity:.3f}"
        )


def test_unusual_inputs(embedder):
    """
    Test how the transformer embedder handles unusual inputs:
    - Empty strings
    - Very short sentences
    - Very long sentences
    - Made-up words
    """
    # Test cases
    unusual_inputs = [
        "",  # Empty string
        "Hi",  # Very short
        "a",  # Single character
        # Long sentence (>100 words)
        "The quick brown fox jumped over the lazy dog. " * 20,
        # Sentence with made-up words
        "The flibberflop zorkled across the blorsht meadow.",
        # Mix of real and made-up words
        "The programmer debugged the flibberflop algorithm.",
        # Normal sentence for comparison
        "This is a normal English sentence.",
    ]

    # Test 1: All inputs should produce embeddings without errors
    for text in unusual_inputs:
        embedding = embedder.embed(text)
        assert isinstance(embedding, np.ndarray), f"Failed to embed: {text}"
        assert embedding.shape[0] == 384, f"Wrong embedding dimension for: {text}"
        assert not np.all(embedding == 0), f"Zero embedding produced for: {text}"

    # Test 2: Empty/short texts should have low similarity with normal texts
    short_texts = ["", "Hi", "a"]
    normal_text = "This is a normal English sentence."

    for short_text in short_texts:
        similarity = embedder.similarity([short_text], [normal_text])[0][0]
        assert similarity < 0.5, (
            f"Expected low similarity between short and normal text, but got {similarity:.3f}\n"
            f"Short text: '{short_text}'\n"
            f"Normal text: '{normal_text}'"
        )

    # Test 3: Made-up words should produce consistent embeddings
    made_up_text = "The flibberflop zorkled across the blorsht meadow."
    embedding1 = embedder.embed(made_up_text)
    embedding2 = embedder.embed(made_up_text)
    np.testing.assert_array_almost_equal(
        embedding1, embedding2, err_msg="Inconsistent embeddings for made-up words"
    )

    # Test 4: Very long text should produce valid embeddings
    long_text = "The quick brown fox jumped over the lazy dog. " * 20
    embedding = embedder.embed(long_text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] == 384
    assert not np.all(embedding == 0)

    # Test 5: Similar sentences with made-up words should have higher similarity
    made_up_pairs = [
        (
            "The flibberflop crashed the system",
            "The flibberflop caused an error",
            "The weather is nice today",
        )
    ]

    for text1, text2, unrelated in made_up_pairs:
        sim_related = embedder.similarity([text1], [text2])[0][0]
        sim_unrelated = embedder.similarity([text1], [unrelated])[0][0]

        assert sim_related > sim_unrelated, (
            f"Expected higher similarity between related made-up word sentences\n"
            f"Text 1: {text1}\n"
            f"Text 2: {text2}\n"
            f"Unrelated: {unrelated}\n"
            f"Related similarity: {sim_related:.3f}\n"
            f"Unrelated similarity: {sim_unrelated:.3f}"
        )
