from typing import List


class TextPreprocessor:
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size

    def split_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks of approximately equal size"""
        if not text:
            return []

        # For text without sentence breaks, split by chunk_size directly
        if text.strip() and all(c == text[0] for c in text):
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

        # Split into sentences
        sentences = [s.strip() + "." for s in text.split(".") if s.strip()]

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If adding this sentence would exceed chunk size, save current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

            # If single sentence is longer than chunk size, split it
            if sentence_length > self.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # Split long sentence into smaller pieces
                words = sentence.split()
                current_piece = []
                current_piece_length = 0

                for word in words:
                    if current_piece_length + len(word) + 1 > self.chunk_size:
                        chunks.append(" ".join(current_piece))
                        current_piece = [word]
                        current_piece_length = len(word)
                    else:
                        current_piece.append(word)
                        current_piece_length += len(word) + 1

                if current_piece:
                    chunks.append(" ".join(current_piece))
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # Add any remaining text
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
