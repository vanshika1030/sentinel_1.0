"""Embedding generation with swappable backends.

Provides an abstract base class and an OpenAI implementation using
LangChain's embedding wrappers. Supports batch embedding.
"""

from __future__ import annotations

import abc
import logging

from langchain_openai import OpenAIEmbeddings

from sentinel.core.config import settings

logger = logging.getLogger(__name__)


class BaseEmbedder(abc.ABC):
    """Abstract base class for embedding providers."""

    @abc.abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """

    @abc.abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings.

        Args:
            texts: A list of texts to embed.

        Returns:
            A list of embedding vectors.
        """

    @abc.abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        """Embed a query string (may use a different model/prefix).

        Args:
            query: The query text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embedding provider using LangChain."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        """Initialize the OpenAI embedder.

        Args:
            model: The OpenAI embedding model name. Defaults to config.
            api_key: OpenAI API key. Defaults to config.
            batch_size: Max texts per batch API call. Defaults to config.
        """
        self._model = model or settings.embedding_model
        self._api_key = api_key or settings.openai_api_key or None
        self._batch_size = batch_size or settings.embedding_batch_size
        self._embeddings = OpenAIEmbeddings(
            model=self._model,
            dimensions=settings.embedding_dimensions,
            openai_api_key=self._api_key,
        )

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        result = await self._embeddings.aembed_documents([text])
        return result[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, respecting batch size limits."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            embeddings = await self._embeddings.aembed_documents(batch)
            all_embeddings.extend(embeddings)
            logger.debug("Embedded batch %d-%d of %d", i, i + len(batch), len(texts))
        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        """Embed a query string."""
        return await self._embeddings.aembed_query(query)
