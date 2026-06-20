"""GitHub source — fetches public repo info via REST API.

Extracts repo description, README content, and latest releases
from public GitHub repositories without authentication.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from sentinel.core.config import settings
from sentinel.ingestion.sources.base_source import BaseSource, FetchResult

logger = logging.getLogger(__name__)


class GitHubSource(BaseSource):
    """Fetches public GitHub repository information."""

    source_type = "github"
    API_BASE = "https://api.github.com"

    @staticmethod
    def _parse_repo_path(url: str) -> str | None:
        """Extract owner/repo from a GitHub URL.

        Args:
            url: GitHub repository URL.

        Returns:
            'owner/repo' string, or None if parsing fails.
        """
        # Handle: https://github.com/owner/repo
        parts = url.rstrip("/").split("github.com/")
        if len(parts) < 2:
            return None
        path = parts[1]
        segments = path.split("/")
        if len(segments) >= 2:
            return f"{segments[0]}/{segments[1]}"
        return None

    async def _fetch_json(self, client: httpx.AsyncClient, url: str) -> Any:
        """Fetch JSON from GitHub API.

        Args:
            client: httpx async client.
            url: API URL.

        Returns:
            Parsed JSON response.
        """
        response = await client.get(url)
        response.raise_for_status()
        return response.json()

    async def fetch(self, url: str) -> FetchResult:
        """Fetch public GitHub repo info.

        Extracts description, README, and latest 5 releases.

        Args:
            url: GitHub repository URL.

        Returns:
            FetchResult with combined repo information.
        """
        repo_path = self._parse_repo_path(url)
        if not repo_path:
            return FetchResult(
                url=url, success=False,
                error="Could not parse GitHub repo path",
                source_type=self.source_type,
            )

        try:
            async with httpx.AsyncClient(
                timeout=settings.http_timeout,
                headers={
                    "Accept": "application/vnd.github.v3+json",
                    "User-Agent": "Sentinel/0.1",
                },
            ) as client:
                # Fetch repo info
                repo_data = await self._fetch_json(
                    client, f"{self.API_BASE}/repos/{repo_path}"
                )

                # Fetch README
                readme_text = ""
                try:
                    readme_resp = await client.get(
                        f"{self.API_BASE}/repos/{repo_path}/readme",
                        headers={"Accept": "application/vnd.github.v3.raw"},
                    )
                    if readme_resp.status_code == 200:
                        readme_text = readme_resp.text
                except Exception:
                    pass

                # Fetch latest releases (last 5)
                releases_text = ""
                try:
                    releases = await self._fetch_json(
                        client,
                        f"{self.API_BASE}/repos/{repo_path}/releases?per_page=5",
                    )
                    release_parts = []
                    for rel in releases[:5]:
                        release_parts.append(
                            f"## {rel.get('name', rel.get('tag_name', 'Unknown'))}\n"
                            f"Tag: {rel.get('tag_name', '')}\n"
                            f"Published: {rel.get('published_at', '')}\n"
                            f"{rel.get('body', '')}\n"
                        )
                    releases_text = "\n---\n".join(release_parts)
                except Exception:
                    pass

            # Combine all content
            description = repo_data.get("description", "") or ""
            stars = repo_data.get("stargazers_count", 0)
            language = repo_data.get("language", "")

            content_parts = [
                f"# {repo_path}",
                f"Description: {description}",
                f"Stars: {stars} | Language: {language}",
                "",
                "## README",
                readme_text,
            ]
            if releases_text:
                content_parts.extend(["", "## Latest Releases", releases_text])

            content = "\n".join(content_parts)

            return FetchResult(
                url=url,
                title=f"{repo_path} — GitHub",
                content=content,
                meta_description=description,
                source_type=self.source_type,
                metadata={
                    "stars": stars,
                    "language": language,
                    "repo_path": repo_path,
                },
            )

        except Exception as exc:
            logger.error("Error fetching GitHub repo %s: %s", url, exc)
            return FetchResult(
                url=url, success=False, error=str(exc),
                source_type=self.source_type,
            )
