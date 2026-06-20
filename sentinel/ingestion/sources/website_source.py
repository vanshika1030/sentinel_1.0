"""Website source — fetches and parses HTML pages.

Uses httpx for async HTTP requests and BeautifulSoup for
HTML parsing and text extraction.
"""

from __future__ import annotations

import logging

import httpx
from bs4 import BeautifulSoup

from sentinel.core.config import settings
from sentinel.ingestion.sources.base_source import BaseSource, FetchResult

logger = logging.getLogger(__name__)


class WebsiteSource(BaseSource):
    """Fetches and parses website HTML pages."""

    source_type = "website"

    async def fetch(self, url: str) -> FetchResult:
        """Fetch a web page and extract its text content.

        Args:
            url: The page URL to fetch.

        Returns:
            FetchResult with extracted title, text, and meta description.
        """
        try:
            async with httpx.AsyncClient(
                timeout=settings.http_timeout,
                follow_redirects=True,
                headers={"User-Agent": "Sentinel/0.1 (Competitive Intelligence Bot)"},
            ) as client:
                response = await client.get(url)
                response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()

            # Extract title
            title = ""
            if soup.title and soup.title.string:
                title = soup.title.string.strip()

            # Extract meta description
            meta_desc = ""
            meta_tag = soup.find("meta", attrs={"name": "description"})
            if meta_tag and meta_tag.get("content"):
                meta_desc = meta_tag["content"].strip()

            # Extract main text content
            # Try to find main content area first
            main = soup.find("main") or soup.find("article") or soup.find(
                "div", class_="content"
            )
            if main:
                text = main.get_text(separator="\n", strip=True)
            else:
                text = soup.get_text(separator="\n", strip=True)

            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = "\n".join(lines)

            return FetchResult(
                url=url,
                title=title,
                content=text,
                meta_description=meta_desc,
                source_type=self.source_type,
                metadata={"status_code": response.status_code},
            )

        except httpx.HTTPStatusError as exc:
            logger.error("HTTP %d for %s", exc.response.status_code, url)
            return FetchResult(
                url=url, success=False,
                error=f"HTTP {exc.response.status_code}",
                source_type=self.source_type,
            )
        except httpx.TimeoutException:
            logger.error("Timeout fetching %s", url)
            return FetchResult(
                url=url, success=False, error="Timeout",
                source_type=self.source_type,
            )
        except httpx.ConnectError:
            logger.error("Connection error for %s", url)
            return FetchResult(
                url=url, success=False, error="Connection error",
                source_type=self.source_type,
            )
        except Exception as exc:
            logger.error("Unexpected error fetching %s: %s", url, exc)
            return FetchResult(
                url=url, success=False, error=str(exc),
                source_type=self.source_type,
            )
