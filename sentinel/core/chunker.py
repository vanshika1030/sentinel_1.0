"""Text chunking using LangChain's RecursiveCharacterTextSplitter.

Provides a factory class for creating text splitters with configurable
chunk sizes and overlaps.
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentinel.core.config import settings


class ChunkerFactory:
    """Factory for creating text chunkers.

    Wraps LangChain's RecursiveCharacterTextSplitter with sensible defaults
    loaded from config, while allowing per-call overrides.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
    ) -> None:
        """Initialize the chunker factory.

        Args:
            chunk_size: Maximum characters per chunk. Defaults to config value.
            chunk_overlap: Overlap between consecutive chunks. Defaults to config value.
            separators: Custom separator hierarchy. Defaults to LangChain's built-in.
        """
        self._chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self._chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        self._separators = separators
        self._splitter: RecursiveCharacterTextSplitter | None = None

    def create_splitter(self) -> RecursiveCharacterTextSplitter:
        """Create and return a configured text splitter instance."""
        if self._splitter is None:
            kwargs: dict = {
                "chunk_size": self._chunk_size,
                "chunk_overlap": self._chunk_overlap,
                "length_function": len,
                "is_separator_regex": False,
            }
            if self._separators:
                kwargs["separators"] = self._separators
            self._splitter = RecursiveCharacterTextSplitter(**kwargs)
        return self._splitter

    def split_text(self, text: str) -> list[str]:
        """Split text into chunks.

        Args:
            text: The input text to split.

        Returns:
            A list of text chunks.
        """
        splitter = self.create_splitter()
        return splitter.split_text(text)

    def split_documents(self, documents: list) -> list:
        """Split LangChain Document objects into chunks.

        Args:
            documents: List of LangChain Document objects.

        Returns:
            A list of chunked Document objects.
        """
        splitter = self.create_splitter()
        return splitter.split_documents(documents)
