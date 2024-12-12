from typing import List


class TextPreprocessor:
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size

    def split_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks of approximately equal size without NLTK"""
        # Simple split by periods as a basic sentence splitter
        sentences = [s.strip() + "." for s in text.split(".") if s.strip()]

        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            if current_size + len(sentence) > self.chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_size = len(sentence)
            else:
                current_chunk.append(sentence)
                current_size += len(sentence)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
