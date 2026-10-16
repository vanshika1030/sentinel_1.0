"""Hybrid retriever with vector search, BM25, RRF fusion, and cross-encoder re-ranking.

Combines Qdrant vector search with BM25 keyword search, merges results
via Reciprocal Rank Fusion, and re-ranks with a cross-encoder model.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field, replace
from typing import Any

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from sentinel.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    """A document retrieved from the search pipeline."""

    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


class HybridRetriever:
    """Hybrid retrieval pipeline: vector + BM25 + RRF + cross-encoder.

    Steps:
        1. Qdrant vector search → top K candidates
        2. BM25 keyword search → top K candidates
        3. Reciprocal Rank Fusion to merge both ranked lists
        4. Cross-encoder re-ranking → final top N
    """

    def __init__(
        self,
        top_k: int | None = None,
        reranker_top_k: int | None = None,
        reranker_model: str | None = None,
        rrf_k: int | None = None,
    ) -> None:
        """Initialize the hybrid retriever.

        Args:
            top_k: Number of candidates from each search method.
            reranker_top_k: Number of final results after re-ranking.
            reranker_model: Cross-encoder model name for re-ranking.
            rrf_k: RRF constant (higher = more weight to lower ranks).
        """
        self._top_k = top_k or settings.retriever_top_k
        self._reranker_top_k = reranker_top_k or settings.reranker_top_k
        self._rrf_k = rrf_k if rrf_k is not None else settings.rrf_k
        self._reranker_model_name = reranker_model or settings.reranker_model
        self._cross_encoder: CrossEncoder | None = None

    def _get_cross_encoder(self) -> CrossEncoder:
        """Lazily load the cross-encoder model."""
        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoder(self._reranker_model_name)
        return self._cross_encoder

    @staticmethod
    def reciprocal_rank_fusion(
        ranked_lists: list[list[RetrievedDocument]],
        k: int = 60,
    ) -> list[RetrievedDocument]:
        """Merge multiple ranked lists using Reciprocal Rank Fusion.

        Args:
            ranked_lists: A list of ranked document lists.
            k: RRF constant controlling rank sensitivity.

        Returns:
            A single merged and re-scored list of documents.
        """
        scores: dict[str, float] = {}
        doc_map: dict[str, RetrievedDocument] = {}

        for ranked_list in ranked_lists:
            for rank, doc in enumerate(ranked_list):
                rrf_score = 1.0 / (k + rank + 1)
                scores[doc.doc_id] = scores.get(doc.doc_id, 0.0) + rrf_score
                if doc.doc_id not in doc_map:
                    doc_map[doc.doc_id] = doc

        # Sort by fused score descending
        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        result = []
        for doc_id in sorted_ids:
            doc = doc_map[doc_id]
            result.append(replace(doc, score=scores[doc_id]))
        return result

    def bm25_search(
        self,
        query: str,
        corpus: list[RetrievedDocument],
        top_k: int | None = None,
    ) -> list[RetrievedDocument]:
        """Run BM25 keyword search over a corpus of documents.

        Args:
            query: The search query.
            corpus: List of documents to search.
            top_k: Number of top results to return.

        Returns:
            Top-K documents ranked by BM25 score.
        """
        top_k = top_k or self._top_k
        if not corpus:
            return []

        tokenized_corpus = [doc.text.lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        scored_docs = list(zip(corpus, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc, score in scored_docs[:top_k]:
            results.append(replace(doc, score=float(score)))
        return results

    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int | None = None,
    ) -> list[RetrievedDocument]:
        """Re-rank documents using a cross-encoder model.

        Args:
            query: The original query.
            documents: Candidate documents to re-rank.
            top_k: Number of top results to return.

        Returns:
            Top-K documents re-ranked by cross-encoder score.
        """
        top_k = top_k or self._reranker_top_k
        if not documents:
            return []

        encoder = self._get_cross_encoder()
        pairs = [[query, doc.text] for doc in documents]
        scores = encoder.predict(pairs)

        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc, score in scored_docs[:top_k]:
            results.append(replace(doc, score=float(score)))
        return results

    async def retrieve(
        self,
        query: str,
        vector_results: list[RetrievedDocument],
        corpus: list[RetrievedDocument] | None = None,
    ) -> list[RetrievedDocument]:
        """Full hybrid retrieval pipeline.

        Args:
            query: The user's natural language query.
            vector_results: Documents from vector search (Qdrant).
            corpus: Full corpus for BM25 search. If None, uses vector_results.

        Returns:
            Final top-K re-ranked documents.
        """
        bm25_corpus = corpus if corpus is not None else vector_results
        bm25_results = self.bm25_search(query, bm25_corpus, self._top_k)

        # Reciprocal Rank Fusion
        fused = self.reciprocal_rank_fusion(
            [vector_results[: self._top_k], bm25_results],
            k=self._rrf_k,
        )

        # Cross-encoder re-ranking
        reranked = await asyncio.to_thread(self.rerank, query, fused, self._reranker_top_k)
        logger.info(
            "Hybrid retrieval: %d vector + %d BM25 → %d fused → %d reranked",
            len(vector_results),
            len(bm25_results),
            len(fused),
            len(reranked),
        )
        return reranked
