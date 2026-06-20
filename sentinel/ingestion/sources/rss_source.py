"""RSS/Atom source — parses syndication feeds.

Uses feedparser to extract entries from RSS and Atom feeds,
including title, summary, and published date.
"""

from __future__ import annotations

import logging
from typing import Any

import feedparser
import httpx

from sentinel.core.config import settings
from sentinel.ingestion.sources.base_source import BaseSource, FetchResult

logger = logging.getLogger(__name__)


class RSSSource(BaseSource):
    """Fetches and parses RSS/Atom feeds."""

    source_type = "rss"

    async def fetch(self, url: str) -> FetchResult:
        """Fetch and parse an RSS/Atom feed.

        Extracts title, summary, and published date for each entry.

        Args:
            url: The feed URL.

        Returns:
            FetchResult with combined feed entries as text.
        """
        try:
            # Fetch the feed content
            async with httpx.AsyncClient(
                timeout=settings.http_timeout,
                follow_redirects=True,
                headers={"User-Agent": "Sentinel/0.1 (Feed Reader)"},
            ) as client:
                response = await client.get(url)
                response.raise_for_status()

            feed = feedparser.parse(response.text)

            if feed.bozo and not feed.entries:
                return FetchResult(
                    url=url, success=False,
                    error="Failed to parse feed",
                    source_type=self.source_type,
                )

            feed_title = feed.feed.get("title", "Unknown Feed")
            entries_text = []

            for entry in feed.entries:
                title = entry.get("title", "Untitled")
                summary = entry.get("summary", entry.get("description", ""))
                published = entry.get("published", entry.get("updated", ""))
                link = entry.get("link", "")

                entries_text.append(
                    f"## {title}\n"
                    f"Published: {published}\n"
                    f"Link: {link}\n"
                    f"{summary}\n"
                )

            content = f"# {feed_title}\n\n" + "\n---\n".join(entries_text)

            return FetchResult(
                url=url,
                title=feed_title,
                content=content,
                source_type=self.source_type,
                metadata={"entry_count": len(feed.entries)},
            )

        except Exception as exc:
            logger.error("Error fetching RSS feed %s: %s", url, exc)
            return FetchResult(
                url=url, success=False, error=str(exc),
                source_type=self.source_type,
            )
