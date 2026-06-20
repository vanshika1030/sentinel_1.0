"""Change significance scorer — combines multiple signals.

Scores changes on a 0.0-1.0 scale using:
  - Diff ratio (how much changed) — weight 0.3
  - Semantic similarity of old vs new embeddings — weight 0.4
  - Keyword trigger matches — weight 0.3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sentinel.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ScoreResult:
    """Result of significance scoring."""

    score: float
    signals_breakdown: dict[str, float] = field(default_factory=dict)
    is_significant: bool = False
    triggered_keywords: list[str] = field(default_factory=list)


class ChangeScorer:
    """Scores the significance of content changes."""

    WEIGHT_DIFF = 0.3
    WEIGHT_SEMANTIC = 0.4
    WEIGHT_KEYWORDS = 0.3

    def __init__(
        self,
        keyword_triggers: list[str] | None = None,
        significance_threshold: float | None = None,
    ) -> None:
        """Initialize the change scorer.

        Args:
            keyword_triggers: Keywords that indicate significant changes.
            significance_threshold: Score threshold for significance.
        """
        self._keywords = keyword_triggers or settings.keyword_triggers
        self._threshold = (
            significance_threshold
            if significance_threshold is not None
            else settings.significance_threshold
        )

    def _score_diff_ratio(self, diff_ratio: float) -> float:
        """Score based on how much content changed.

        Args:
            diff_ratio: 0.0 = identical, 1.0 = completely different.

        Returns:
            Normalized score (0.0 to 1.0).
        """
        # Small changes (< 5%) are likely noise
        if diff_ratio < 0.05:
            return 0.0
        # Scale linearly, capping at 1.0
        return min(1.0, diff_ratio * 2.0)

    def _score_semantic_distance(
        self, similarity: float | None
    ) -> float:
        """Score based on semantic distance between old and new content.

        Low similarity = high significance.

        Args:
            similarity: Cosine similarity between embeddings (0.0 to 1.0).

        Returns:
            Normalized significance score.
        """
        if similarity is None:
            return 0.5  # Default when embeddings unavailable
        # Invert: low similarity → high score
        return max(0.0, min(1.0, 1.0 - similarity))

    def _score_keywords(self, diff_text: str) -> tuple[float, list[str]]:
        """Score based on keyword trigger matches in the diff.

        Args:
            diff_text: The unified diff text.

        Returns:
            Tuple of (score, list of triggered keywords).
        """
        diff_lower = diff_text.lower()
        triggered = [
            kw for kw in self._keywords if kw.lower() in diff_lower
        ]
        if not triggered:
            return 0.0, []
        # Score based on how many keywords triggered
        score = min(1.0, len(triggered) / 3.0)
        return score, triggered

    def score(
        self,
        diff_ratio: float,
        diff_text: str,
        semantic_similarity: float | None = None,
    ) -> ScoreResult:
        """Compute the overall significance score.

        Args:
            diff_ratio: How much the content changed (0-1).
            diff_text: The unified diff text.
            semantic_similarity: Cosine similarity of embeddings.

        Returns:
            ScoreResult with the overall score and breakdown.
        """
        diff_score = self._score_diff_ratio(diff_ratio)
        semantic_score = self._score_semantic_distance(semantic_similarity)
        keyword_score, triggered = self._score_keywords(diff_text)

        overall = (
            self.WEIGHT_DIFF * diff_score
            + self.WEIGHT_SEMANTIC * semantic_score
            + self.WEIGHT_KEYWORDS * keyword_score
        )

        result = ScoreResult(
            score=round(overall, 4),
            signals_breakdown={
                "diff_ratio": round(diff_score, 4),
                "semantic_distance": round(semantic_score, 4),
                "keyword_triggers": round(keyword_score, 4),
            },
            is_significant=overall > self._threshold,
            triggered_keywords=triggered,
        )

        logger.debug(
            "Score for change: %.4f (significant=%s, keywords=%s)",
            result.score,
            result.is_significant,
            triggered,
        )
        return result
