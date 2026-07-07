"""Sentence-boundary text splitter for the knowledge-base ingest.

Splits text into sentences (punctuation boundaries, with newlines also
treated as boundaries so one deployment-log line stays one sentence), then
greedily packs sentences into chunks up to a size budget with a small
sentence overlap between consecutive chunks.
"""

import re

SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")


class SentenceSplitter:
    """Packs whole sentences into chunks instead of cutting mid-sentence.

    A single sentence longer than chunk_size is emitted as its own chunk
    rather than being cut.
    """

    def __init__(self, chunk_size: int = 1000, overlap_sentences: int = 1) -> None:
        self.chunk_size = chunk_size
        self.overlap_sentences = overlap_sentences

    def split_text(self, text: str) -> list[str]:
        """Split text into sentence-packed chunks within the size budget."""
        sentences = [
            sentence.strip()
            for sentence in SENTENCE_BOUNDARY_PATTERN.split(text)
            if sentence and sentence.strip()
        ]
        chunks: list[str] = []
        current_sentences: list[str] = []
        current_length = 0
        for sentence in sentences:
            if current_sentences and (
                current_length + len(sentence) + 1 > self.chunk_size
            ):
                chunks.append(" ".join(current_sentences))
                current_sentences = current_sentences[-self.overlap_sentences :]
                current_length = sum(len(kept) + 1 for kept in current_sentences)
            current_sentences.append(sentence)
            current_length += len(sentence) + 1
        if current_sentences:
            chunks.append(" ".join(current_sentences))
        return chunks
