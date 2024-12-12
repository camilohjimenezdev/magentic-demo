import requests
import re
from bs4 import BeautifulSoup
from typing import List, Dict


class GutenbergDownloader:
    def __init__(self, base_url: str = "https://www.gutenberg.org/files/"):
        self.base_url = base_url

    def download_book(self, book_id):
        # Construct the URL
        url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"

        # Send a GET request to the URL
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Get the content of the book
            book_content = response.text

            # Clean the content (remove headers and footers)
            clean_content = self.strip_headers(book_content)

            return clean_content
        else:
            return f"Failed to download book with ID {book_id}"

    def strip_headers(self, text):
        # Find the start of the book content
        start_match = re.search(
            r"\*\*\* START OF THIS PROJECT GUTENBERG EBOOK .+? \*\*\*", text
        )
        if start_match:
            start = start_match.end()
        else:
            start = 0

        # Find the end of the book content
        end_match = re.search(
            r"\*\*\* END OF THIS PROJECT GUTENBERG EBOOK .+? \*\*\*", text
        )
        if end_match:
            end = end_match.start()
        else:
            end = len(text)

        # Return the cleaned content
        return text[start:end].strip()


# main function to download a book
if __name__ == "__main__":
    downloader = GutenbergDownloader()
    book_id = 2701  # Moby Dick
    book_content = downloader.download_book(book_id)

    if not book_content.startswith("Failed"):
        print(
            f"Book downloaded successfully. First 200 characters:\n{book_content[:200]}"
        )

        # Optionally, save the book to a file
        with open(f"book_{book_id}.txt", "w", encoding="utf-8") as file:
            file.write(book_content)
        print(f"Book saved as book_{book_id}.txt")
    else:
        print(book_content)
