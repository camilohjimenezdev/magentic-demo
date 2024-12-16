"""
Test suite for the GutenbergDownloader class.

This module contains unit tests that verify the functionality of the GutenbergDownloader,
which is responsible for downloading books from Project Gutenberg's digital library.
"""

import pytest
from app.ingestion.downloader import GutenbergDownloader


def test_download_book():
    """
    Test successful download of a single book from Project Gutenberg.

    Tests downloading 'Moby Dick' (book_id: 2701) and verifies that:
    - The returned content is a string
    - The content is not empty
    - The content contains the expected book title
    """
    downloader = GutenbergDownloader()
    book_id = 2701  # Moby Dick

    content = downloader.download_book(book_id)

    assert isinstance(content, str)
    assert len(content) > 0
    assert "Moby Dick" in content


def test_invalid_book_id():
    """
    Test handling of invalid book IDs.

    Verifies that the downloader properly handles invalid book IDs
    by returning an error message starting with 'Failed'.
    """
    downloader = GutenbergDownloader()
    book_id = -1

    content = downloader.download_book(book_id)
    assert content.startswith("Failed")


@pytest.mark.skip(
    reason="This test is skipped by default to avoid hitting Gutenberg API too frequently"
)
def test_multiple_downloads():
    """
    Test downloading multiple books in succession.

    Attempts to download three classic books:
    - Moby Dick (2701)
    - Pride and Prejudice (1342)
    - Frankenstein (84)

    This test is skipped by default to avoid overwhelming the Gutenberg API.

    Verifies that each download:
    - Returns a string
    - Contains actual content
    """
    downloader = GutenbergDownloader()
    book_ids = [2701, 1342, 84]  # Moby Dick, Pride and Prejudice, Frankenstein

    for book_id in book_ids:
        content = downloader.download_book(book_id)
        assert isinstance(content, str)
        assert len(content) > 0
