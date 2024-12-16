from app.ingestion.preprocessor import TextPreprocessor

"""
Test module for TextPreprocessor class.
This module contains unit tests to verify the text chunking functionality
of the TextPreprocessor class.
"""


def test_split_into_chunks():
    """
    Test the basic functionality of splitting text into chunks.

    Verifies that:
    1. The text is successfully split into chunks
    2. Each chunk respects the maximum size limit
    3. All chunks are strings

    The test uses a simple multi-sentence text with a chunk size of 100 characters.
    """
    preprocessor = TextPreprocessor(chunk_size=100)
    text = "This is a test sentence. This is another test sentence. And here is a third one."

    chunks = preprocessor.split_into_chunks(text)

    assert len(chunks) > 0
    assert all(len(chunk) <= 100 for chunk in chunks)
    assert all(isinstance(chunk, str) for chunk in chunks)


def test_empty_text():
    """
    Test handling of empty text input.

    Verifies that the preprocessor returns an empty list when given an empty string,
    ensuring proper handling of edge cases.
    """
    preprocessor = TextPreprocessor()
    chunks = preprocessor.split_into_chunks("")
    assert chunks == []


def test_chunk_size_respected():
    """
    Test that the specified chunk size is strictly enforced.

    Creates a test string of 100 'A' characters and verifies that each resulting
    chunk is no larger than the specified size of 50 characters.
    """
    preprocessor = TextPreprocessor(chunk_size=50)
    text = "A" * 100
    chunks = preprocessor.split_into_chunks(text)
    assert all(len(chunk) <= 50 for chunk in chunks)
