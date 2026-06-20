"""Base source — abstract interface for content extraction.

All source types (website, GitHub, RSS) inherit from this base
and implement the fetch() method.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FetchResult:
    """Result of a source fetch operation."""

    url: str
    title: str = ""
    content: str = ""
    meta_description: str = ""
    source_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str = ""


class BaseSource(abc.ABC):
    """Abstract base class for content sources."""

    source_type: str = "base"

    @abc.abstractmethod
    async def fetch(self, url: str) -> FetchResult:
        """Fetch and extract content from a URL.

        Args:
            url: The URL to fetch.

        Returns:
            A FetchResult with extracted content.
        """
