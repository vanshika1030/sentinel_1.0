"""Timeline store — persists and queries change records.

Stores detected changes in PostgreSQL with significance scores
and provides flexible querying by competitor, date range, and significance.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

from sqlalchemy import select, desc, and_
from sqlalchemy.ext.asyncio import AsyncSession

from sentinel.models.database import Change, get_session_factory

logger = logging.getLogger(__name__)


class TimelineStore:
    """Manages change timeline records in PostgreSQL."""

    def __init__(self, session: AsyncSession | None = None) -> None:
        """Initialize the timeline store.

        Args:
            session: Optional async session.
        """
        self._session = session

    async def _get_session(self) -> AsyncSession:
        """Get or create an async session."""
        if self._session is None:
            factory = get_session_factory()
            self._session = factory()
        return self._session

    async def save_change(
        self,
        url: str,
        competitor_name: str,
        diff_text: str,
        significance_score: float,
        is_significant: bool,
    ) -> Change:
        """Save a change record.

        Args:
            url: The source URL.
            competitor_name: Competitor name.
            diff_text: The unified diff text.
            significance_score: Computed significance score.
            is_significant: Whether the score exceeds the threshold.

        Returns:
            The created Change ORM object.
        """
        session = await self._get_session()
        change = Change(
            url=url,
            competitor_name=competitor_name,
            diff_text=diff_text,
            significance_score=significance_score,
            is_significant=is_significant,
        )
        session.add(change)
        await session.commit()
        await session.refresh(change)
        logger.debug(
            "Saved change for %s (score=%.4f, significant=%s)",
            url,
            significance_score,
            is_significant,
        )
        return change

    async def get_changes(
        self,
        competitor_name: str | None = None,
        since: datetime.datetime | None = None,
        until: datetime.datetime | None = None,
        significant_only: bool = False,
        limit: int = 100,
    ) -> list[Change]:
        """Query change records with flexible filters.

        Args:
            competitor_name: Filter by competitor name.
            since: Only changes after this datetime.
            until: Only changes before this datetime.
            significant_only: If True, only return significant changes.
            limit: Maximum number of results.

        Returns:
            List of Change objects matching the filters.
        """
        session = await self._get_session()
        conditions = []

        if competitor_name:
            conditions.append(Change.competitor_name == competitor_name)
        if since:
            conditions.append(Change.detected_at >= since)
        if until:
            conditions.append(Change.detected_at <= until)
        if significant_only:
            conditions.append(Change.is_significant == True)  # noqa: E712

        stmt = (
            select(Change)
            .where(and_(*conditions)) if conditions else select(Change)
        )
        stmt = stmt.order_by(desc(Change.detected_at)).limit(limit)

        result = await session.execute(stmt)
        return list(result.scalars().all())
