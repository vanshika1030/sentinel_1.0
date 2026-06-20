"""Database models and engine management.

SQLAlchemy ORM models for snapshots and changes, plus async engine
factory and session management.
"""

from __future__ import annotations

import datetime
from typing import AsyncGenerator

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from sentinel.core.config import settings


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""


class Snapshot(Base):
    """A point-in-time snapshot of fetched content."""

    __tablename__ = "snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    url = Column(String(2048), nullable=False, index=True)
    content_text = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False)
    competitor_name = Column(String(255), nullable=False, index=True)
    fetched_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class Change(Base):
    """A detected change between two snapshots."""

    __tablename__ = "changes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    url = Column(String(2048), nullable=False, index=True)
    competitor_name = Column(String(255), nullable=False, index=True)
    diff_text = Column(Text, nullable=False)
    significance_score = Column(Float, nullable=False, default=0.0)
    is_significant = Column(Boolean, nullable=False, default=False)
    detected_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


# ── Engine and session management ────────────────────────────────

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Get or create the async SQLAlchemy engine.

    Returns:
        The singleton AsyncEngine instance.
    """
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            settings.postgres_url,
            echo=False,
            pool_size=10,
            max_overflow=20,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the session factory.

    Returns:
        The async session factory.
    """
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(), class_=AsyncSession, expire_on_commit=False
        )
    return _session_factory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session.

    Yields:
        An AsyncSession for database operations.
    """
    factory = get_session_factory()
    async with factory() as session:
        yield session


async def init_db() -> None:
    """Create all database tables."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Close the database engine."""
    global _engine, _session_factory
    if _engine:
        await _engine.dispose()
        _engine = None
        _session_factory = None
