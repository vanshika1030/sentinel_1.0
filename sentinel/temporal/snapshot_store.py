"""Snapshot store — persists page snapshots in PostgreSQL.

Stores content snapshots with hashes for change detection.
"""

from __future__ import annotations

import datetime
import hashlib
import logging
from typing import Any

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from sentinel.models.database import Snapshot, get_session_factory

logger = logging.getLogger(__name__)


class SnapshotStore:
    """Manages content snapshots in PostgreSQL."""

    def __init__(self, session: AsyncSession | None = None) -> None:
        """Initialize the snapshot store.

        Args:
            session: Optional async session (creates one if not provided).
        """
        self._session = session

    async def _get_session(self) -> AsyncSession:
        """Get or create an async session."""
        if self._session is None:
            factory = get_session_factory()
            self._session = factory()
        return self._session

    async def save_snapshot(
        self,
        url: str,
        content_text: str,
        competitor_name: str,
    ) -> Snapshot:
        """Save a new content snapshot.

        Args:
            url: The source URL.
            content_text: The fetched text content.
            competitor_name: Name of the competitor.

        Returns:
            The created Snapshot ORM object.
        """
        session = await self._get_session()
        content_hash = hashlib.sha256(content_text.encode("utf-8")).hexdigest()

        snapshot = Snapshot(
            url=url,
            content_text=content_text,
            content_hash=content_hash,
            competitor_name=competitor_name,
        )
        session.add(snapshot)
        await session.commit()
        await session.refresh(snapshot)
        logger.debug("Saved snapshot for %s (hash: %s...)", url, content_hash[:12])
        return snapshot

    async def get_previous_snapshot(self, url: str) -> Snapshot | None:
        """Get the most recent previous snapshot for a URL.

        Args:
            url: The source URL.

        Returns:
            The most recent Snapshot, or None if no previous snapshot exists.
        """
        session = await self._get_session()
        stmt = (
            select(Snapshot)
            .where(Snapshot.url == url)
            .order_by(desc(Snapshot.fetched_at))
            .limit(1)
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_snapshots_since(
        self,
        url: str,
        since: datetime.datetime,
    ) -> list[Snapshot]:
        """Get all snapshots for a URL since a given date.

        Args:
            url: The source URL.
            since: Fetch snapshots after this datetime.

        Returns:
            List of Snapshot objects.
        """
        session = await self._get_session()
        stmt = (
            select(Snapshot)
            .where(Snapshot.url == url, Snapshot.fetched_at >= since)
            .order_by(Snapshot.fetched_at)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())
