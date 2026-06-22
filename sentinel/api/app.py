"""FastAPI application — main entry point.

Configures the app with CORS, lifespan management, global error
handling, and all route registrations.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import redis.asyncio as aioredis
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from sentinel.core.config import settings

logger = logging.getLogger(__name__)

# Global app state for dependency injection
_app_state: dict[str, Any] = {}


def get_app_state() -> dict[str, Any]:
    """Return the global app state dictionary."""
    return _app_state


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan — initialize and cleanup resources."""
    global _app_state
    logger.info("Sentinel starting up...")

    redis_client = None
    try:
        # Initialize Redis (optional for local dev)
        try:
            redis_client = aioredis.from_url(
                settings.redis_url, decode_responses=True
            )
            await redis_client.ping()
            _app_state["redis"] = redis_client
            logger.info("Redis connected")
        except Exception as e:
            logger.warning(f"Redis unavailable (optional): {e}")

        # Initialize database
        from sentinel.models.database import get_engine, init_db
        await init_db()
        _app_state["db_engine"] = get_engine()
        logger.info("Database initialized")

        # Initialize shard manager (optional for local dev)
        try:
            from sentinel.sharding.shard_manager import ShardManager
            shard_manager = ShardManager()
            await shard_manager.ensure_collection()
            _app_state["shard_manager"] = shard_manager
            logger.info("Shard manager initialized")
        except Exception as e:
            logger.warning(f"Shard manager unavailable (optional): {e}")
            shard_manager = None

        # Initialize core components (optional if dependencies unavailable)
        try:
            from sentinel.core.embedder import OpenAIEmbedder
            from sentinel.core.generator import RAGGenerator
            from sentinel.core.retriever import HybridRetriever
            from sentinel.sharding.scatter_gather import ScatterGather
            from sentinel.sharding.shard_router import ShardRouter

            _app_state["embedder"] = OpenAIEmbedder()
            _app_state["retriever"] = HybridRetriever()
            _app_state["generator"] = RAGGenerator()
            if shard_manager:
                _app_state["scatter_gather"] = ScatterGather(shard_manager)
                _app_state["shard_router"] = ShardRouter(shard_manager)
            logger.info("Core components initialized")
        except Exception as e:
            logger.warning(f"Core components partially unavailable: {e}")

        # Initialize temporal components (optional)
        try:
            from sentinel.temporal.snapshot_store import SnapshotStore
            from sentinel.temporal.timeline import TimelineStore

            _app_state["snapshot_store"] = SnapshotStore()
            _app_state["timeline_store"] = TimelineStore()
            logger.info("Temporal components initialized")
        except Exception as e:
            logger.warning(f"Temporal components unavailable: {e}")

        # Initialize ingestion components (optional if Redis unavailable)
        if redis_client:
            try:
                from sentinel.ingestion.producer import JobProducer
                from sentinel.ingestion.scheduler import CrawlScheduler

                producer = JobProducer(redis_client)
                _app_state["producer"] = producer
                _app_state["scheduler"] = CrawlScheduler(producer)
                logger.info("Ingestion components initialized")
            except Exception as e:
                logger.warning(f"Ingestion components unavailable: {e}")

        # Initialize health checker (optional)
        try:
            from sentinel.resilience.health import HealthChecker
            _app_state["health_checker"] = HealthChecker(
                redis_client=redis_client,
                shard_manager=shard_manager if shard_manager else None,
                db_engine=get_engine(),
            )
        except Exception as e:
            logger.warning(f"Health checker unavailable: {e}")

        logger.info("Sentinel startup complete (some optional services may be unavailable)")
        yield

    finally:
        logger.info("Sentinel shutting down...")

        # Cleanup Redis
        if redis_client:
            try:
                await redis_client.close()
            except Exception as e:
                logger.warning(f"Error closing Redis: {e}")

        # Cleanup database
        from sentinel.models.database import close_db
        await close_db()

        _app_state.clear()
        logger.info("Sentinel shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Sentinel",
        description="Distributed Real-Time Competitive Intelligence Engine",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.error("Unhandled exception: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    # Register routes
    from sentinel.api.routes.query import router as query_router
    from sentinel.api.routes.competitors import router as competitors_router
    from sentinel.api.routes.ingest import router as ingest_router
    from sentinel.api.routes.health import router as health_router

    app.include_router(query_router)
    app.include_router(competitors_router)
    app.include_router(ingest_router)
    app.include_router(health_router)

    return app


# Create the app instance
app = create_app()
