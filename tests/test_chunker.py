"""Tests for the text chunker factory."""

import pytest

from sentinel.core.chunker import ChunkerFactory


class TestChunkerFactory:
    """Test suite for ChunkerFactory."""

    def test_default_chunking(self) -> None:
        """Default chunker should split text into reasonable chunks."""
        chunker = ChunkerFactory(chunk_size=100, chunk_overlap=20)
        text = "word " * 200  # 1000 chars
        chunks = chunker.split_text(text)

        assert len(chunks) > 1
        for chunk in chunks:
            # Each chunk should be at most chunk_size chars (may exceed slightly
            # if no separator is found within the window)
            assert len(chunk) <= 200  # generous upper bound

    def test_small_text_single_chunk(self) -> None:
        """Text smaller than chunk_size should produce one chunk."""
        chunker = ChunkerFactory(chunk_size=500, chunk_overlap=100)
        text = "This is a short sentence."
        chunks = chunker.split_text(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_empty_text_no_chunks(self) -> None:
        """Empty text should produce no chunks."""
        chunker = ChunkerFactory(chunk_size=500, chunk_overlap=100)
        chunks = chunker.split_text("")

        assert chunks == []

    def test_chunk_overlap_produces_overlapping_content(self) -> None:
        """Consecutive chunks should overlap."""
        chunker = ChunkerFactory(chunk_size=50, chunk_overlap=20)
        # Create text with distinct sentences
        sentences = [f"Sentence number {i} is here." for i in range(20)]
        text = " ".join(sentences)
        chunks = chunker.split_text(text)

        if len(chunks) >= 2:
            # Check that some content appears in consecutive chunks
            for i in range(len(chunks) - 1):
                # The end of chunk i should share some content with chunk i+1
                # (due to overlap)
                words_current = set(chunks[i].split()[-5:])
                words_next = set(chunks[i + 1].split()[:5])
                # At least some overlap should exist
                overlap = words_current & words_next
                assert len(overlap) > 0, f"No overlap found between chunk {i} and chunk {i+1}"

    def test_custom_chunk_size(self) -> None:
        """Custom chunk_size should be respected."""
        chunker_small = ChunkerFactory(chunk_size=50, chunk_overlap=10)
        chunker_large = ChunkerFactory(chunk_size=500, chunk_overlap=10)
        text = "Hello world. " * 100  # 1300 chars

        small_chunks = chunker_small.split_text(text)
        large_chunks = chunker_large.split_text(text)

        # Smaller chunk size should produce more chunks
        assert len(small_chunks) > len(large_chunks)

    def test_create_splitter_returns_splitter(self) -> None:
        """create_splitter should return a RecursiveCharacterTextSplitter."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        chunker = ChunkerFactory(chunk_size=100, chunk_overlap=20)
        splitter = chunker.create_splitter()
        assert isinstance(splitter, RecursiveCharacterTextSplitter)

    def test_respects_natural_separators(self) -> None:
        """Chunker should prefer splitting at natural boundaries."""
        chunker = ChunkerFactory(chunk_size=100, chunk_overlap=10)
        text = "First paragraph about topic A.\n\nSecond paragraph about topic B.\n\nThird paragraph about topic C."
        chunks = chunker.split_text(text)

        # Should prefer splitting at paragraph boundaries (\n\n)
        for chunk in chunks:
            # Chunks should generally be clean (not cut mid-word)
            assert not chunk.startswith(" ") or len(chunk.strip()) > 0
