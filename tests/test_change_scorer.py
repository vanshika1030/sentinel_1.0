"""Tests for the change significance scorer."""

import pytest

from sentinel.temporal.change_scorer import ChangeScorer, ScoreResult


class TestChangeScorer:
    """Test suite for ChangeScorer."""

    @pytest.fixture
    def scorer(self) -> ChangeScorer:
        """Create a scorer with default settings."""
        return ChangeScorer(
            keyword_triggers=["pricing", "new", "launch", "deprecat",
                              "remov", "feature", "update", "breaking"],
            significance_threshold=0.5,
        )

    def test_cosmetic_change_is_not_significant(self, scorer: ChangeScorer) -> None:
        """Tiny changes with no keywords should score low."""
        result = scorer.score(
            diff_ratio=0.02,  # Very small change
            diff_text="- old timestamp\n+ new timestamp",
            semantic_similarity=0.98,  # Very similar semantically
        )
        assert result.score < 0.5
        assert not result.is_significant
        assert len(result.triggered_keywords) == 0

    def test_major_content_change_is_significant(self, scorer: ChangeScorer) -> None:
        """Large content changes with keywords should score high."""
        result = scorer.score(
            diff_ratio=0.5,  # Major change
            diff_text="+ New feature: We're launching a new pricing plan",
            semantic_similarity=0.3,  # Very different semantically
        )
        assert result.score > 0.5
        assert result.is_significant

    def test_keyword_triggers_boost_score(self, scorer: ChangeScorer) -> None:
        """Keyword matches should boost the significance score."""
        # Same diff ratio and semantic similarity, but one has keywords
        without_keywords = scorer.score(
            diff_ratio=0.3,
            diff_text="+ Some ordinary text without triggers",
            semantic_similarity=0.7,
        )

        with_keywords = scorer.score(
            diff_ratio=0.3,
            diff_text="+ New feature launch with updated pricing",
            semantic_similarity=0.7,
        )

        assert with_keywords.score > without_keywords.score
        assert len(with_keywords.triggered_keywords) > 0

    def test_all_keyword_triggers_detected(self, scorer: ChangeScorer) -> None:
        """All matching keywords should be reported."""
        diff = "+ Breaking change: removing the deprecated feature and updating pricing for the new launch"
        result = scorer.score(
            diff_ratio=0.5,
            diff_text=diff,
            semantic_similarity=0.5,
        )
        # Should detect multiple keywords
        assert "pricing" in result.triggered_keywords
        assert "breaking" in result.triggered_keywords
        assert "new" in result.triggered_keywords

    def test_signals_breakdown_present(self, scorer: ChangeScorer) -> None:
        """Score result should contain a signals breakdown."""
        result = scorer.score(
            diff_ratio=0.3,
            diff_text="+ Some change with a new feature",
            semantic_similarity=0.6,
        )
        assert "diff_ratio" in result.signals_breakdown
        assert "semantic_distance" in result.signals_breakdown
        assert "keyword_triggers" in result.signals_breakdown

    def test_none_semantic_similarity_uses_default(self, scorer: ChangeScorer) -> None:
        """When semantic similarity is None, a neutral default should be used."""
        result = scorer.score(
            diff_ratio=0.3,
            diff_text="+ Some change",
            semantic_similarity=None,
        )
        # Should still produce a valid score
        assert 0.0 <= result.score <= 1.0
        # Default semantic distance should be 0.5
        assert result.signals_breakdown["semantic_distance"] == 0.5

    def test_identical_content_scores_zero(self, scorer: ChangeScorer) -> None:
        """Identical content (no diff) should score near zero."""
        result = scorer.score(
            diff_ratio=0.0,
            diff_text="",
            semantic_similarity=1.0,
        )
        assert result.score < 0.1
        assert not result.is_significant

    def test_score_range_is_normalized(self, scorer: ChangeScorer) -> None:
        """Score should always be between 0.0 and 1.0."""
        test_cases = [
            (0.0, "", 1.0),
            (1.0, "new feature launch breaking update pricing", 0.0),
            (0.5, "some change", 0.5),
            (0.01, "tiny", 0.99),
        ]
        for diff_ratio, diff_text, semantic_sim in test_cases:
            result = scorer.score(diff_ratio, diff_text, semantic_sim)
            assert 0.0 <= result.score <= 1.0, (
                f"Score {result.score} out of range for inputs: "
                f"ratio={diff_ratio}, sim={semantic_sim}"
            )

    def test_custom_threshold(self) -> None:
        """Custom significance threshold should be respected."""
        strict = ChangeScorer(significance_threshold=0.9)
        lenient = ChangeScorer(significance_threshold=0.1)

        strict_result = strict.score(0.3, "+ new feature", 0.5)
        lenient_result = lenient.score(0.3, "+ new feature", 0.5)

        # Same score, but different significance
        assert strict_result.score == lenient_result.score
        assert not strict_result.is_significant or lenient_result.is_significant
