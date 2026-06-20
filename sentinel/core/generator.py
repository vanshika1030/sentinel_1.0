"""RAG answer generation using LangChain.

Takes retrieved chunks + user query, formats them into a prompt with
context, sends to the LLM, and returns an answer with source citations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from sentinel.core.config import settings
from sentinel.core.retriever import RetrievedDocument

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Sentinel, a competitive intelligence analyst.
You answer questions about competitors based ONLY on the provided context.
Always cite your sources by referencing the URL and title.
If the context does not contain enough information, say so honestly.
Be concise, specific, and analytical in your responses."""

HUMAN_PROMPT = """Context:
{context}

---
Question: {question}

Provide a detailed answer based on the context above. Cite sources using [Source: URL] format."""


@dataclass
class GeneratedAnswer:
    """An LLM-generated answer with source citations."""

    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    tokens_used: int = 0


class RAGGenerator:
    """Generate answers from retrieved documents using an LLM."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Initialize the generator.

        Args:
            model: LLM model name. Defaults to config.
            api_key: OpenAI API key. Defaults to config.
            temperature: Sampling temperature. Defaults to config.
            max_tokens: Maximum response tokens. Defaults to config.
        """
        self._model = model or settings.llm_model
        self._api_key = api_key or settings.openai_api_key
        self._temperature = temperature if temperature is not None else settings.llm_temperature
        self._max_tokens = max_tokens or settings.llm_max_tokens

        self._llm = ChatOpenAI(
            model=self._model,
            api_key=self._api_key,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", HUMAN_PROMPT),
            ]
        )
        self._chain = self._prompt | self._llm

    @staticmethod
    def _format_context(documents: list[RetrievedDocument]) -> str:
        """Format retrieved documents into context string.

        Args:
            documents: Retrieved documents with text and metadata.

        Returns:
            Formatted context string.
        """
        parts = []
        for i, doc in enumerate(documents, 1):
            url = doc.metadata.get("url", "unknown")
            title = doc.metadata.get("title", "Untitled")
            competitor = doc.metadata.get("competitor_name", "Unknown")
            parts.append(
                f"[Source {i}]\n"
                f"URL: {url}\n"
                f"Title: {title}\n"
                f"Competitor: {competitor}\n"
                f"Content: {doc.text}\n"
            )
        return "\n---\n".join(parts)

    @staticmethod
    def _extract_sources(documents: list[RetrievedDocument]) -> list[dict[str, Any]]:
        """Extract source metadata from documents.

        Args:
            documents: Retrieved documents.

        Returns:
            List of source dictionaries.
        """
        sources = []
        for doc in documents:
            sources.append(
                {
                    "url": doc.metadata.get("url", ""),
                    "title": doc.metadata.get("title", ""),
                    "snippet": doc.text[:200],
                    "relevance_score": doc.score,
                }
            )
        return sources

    async def generate(
        self,
        question: str,
        documents: list[RetrievedDocument],
    ) -> GeneratedAnswer:
        """Generate an answer from retrieved documents.

        Args:
            question: The user's question.
            documents: Retrieved and re-ranked documents.

        Returns:
            A GeneratedAnswer with the LLM response and sources.
        """
        if not documents:
            return GeneratedAnswer(
                answer="I don't have enough context to answer this question. "
                       "No relevant documents were found.",
                sources=[],
                model=self._model,
            )

        context = self._format_context(documents)
        sources = self._extract_sources(documents)

        logger.info("Generating answer with %d context documents", len(documents))
        response = await self._chain.ainvoke(
            {"context": context, "question": question}
        )

        return GeneratedAnswer(
            answer=response.content,
            sources=sources,
            model=self._model,
            tokens_used=response.response_metadata.get("token_usage", {}).get(
                "total_tokens", 0
            ),
        )
