"""Text differ — computes unified diffs between content versions.

Uses Python's difflib to produce human-readable diffs and
quantitative metrics about what changed.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass


@dataclass
class DiffResult:
    """Result of a text diff computation."""

    added_lines: list[str]
    removed_lines: list[str]
    diff_text: str
    diff_ratio: float  # 0.0 = identical, 1.0 = completely different


class TextDiffer:
    """Computes diffs between text content versions."""

    @staticmethod
    def compute_diff(old_text: str, new_text: str) -> DiffResult:
        """Compute a unified diff between old and new text.

        Args:
            old_text: The previous version of the text.
            new_text: The current version of the text.

        Returns:
            A DiffResult with added/removed lines, diff text, and ratio.
        """
        old_lines = old_text.splitlines(keepends=True)
        new_lines = new_text.splitlines(keepends=True)

        # Compute unified diff
        diff = list(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile="previous",
                tofile="current",
                lineterm="",
            )
        )

        # Extract added and removed lines
        added = [line[1:] for line in diff if line.startswith("+") and not line.startswith("+++")]
        removed = [line[1:] for line in diff if line.startswith("-") and not line.startswith("---")]

        # Compute similarity ratio (1 - similarity = diff_ratio)
        matcher = difflib.SequenceMatcher(None, old_text, new_text)
        similarity = matcher.ratio()
        diff_ratio = 1.0 - similarity

        return DiffResult(
            added_lines=added,
            removed_lines=removed,
            diff_text="\n".join(diff),
            diff_ratio=diff_ratio,
        )
