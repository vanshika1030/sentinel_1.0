"""Entry point for running the worker pool as a module.

Usage: python -m sentinel.ingestion.worker_pool
"""

import asyncio
import logging
import signal
import sys
from typing import Any

from sentinel.core.config import settings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def handle_job(data: dict[str, Any]) -> None:
    """Process a crawl job from the Redis Stream.

    Fetches content from the URL, runs change detection,
    and embeds results into the vector database.

    Args:
        data: The job message with url, source_type, competitor_name.
    """
    from sentinel.ingestion.sources.website_source import WebsiteSource
    from sentinel.ingestion.sources.github_source import GitHubSource
    from sentinel.ingestion.sources.rss_source import RSSSource
    from sentinel.ingestion.rate_limiter import DomainRateLimiter
    from sentinel.ingestion.idempotency import IdempotencyGuard
    from sentinel.temporal.change_detector import ChangeDetector
    from sentinel.temporal.change_scorer import ChangeScorer
    from sentinel.temporal.snapshot_store import SnapshotStore
    from sentinel.temporal.timeline import TimelineStore
    from sentinel.core.embedder import OpenAIEmbedder
    from sentinel.sharding.shard_manager import ShardManager
    from sentinel.sharding.shard_router import ShardRouter

    url = data.get("url", "")
    source_type = data.get("source_type", "website")
    competitor_name = data.get("competitor_name", "unknown")

    logger.info("Processing job: %s (%s) for %s", url, source_type, competitor_name)

    # Rate limit
    rate_limiter = DomainRateLimiter()
    await rate_limiter.acquire(url)

    # Select source handler
    sources = {
        "website": WebsiteSource,
        "github": GitHubSource,
        "rss": RSSSource,
    }
    source_cls = sources.get(source_type, WebsiteSource)
    source = source_cls()

    # Fetch content
    result = await source.fetch(url)
    if not result.success:
        logger.error("Fetch failed for %s: %s", url, result.error)
        raise RuntimeError(f"Fetch failed: {result.error}")

    if not result.content:
        logger.warning("No content extracted from %s", url)
        return

    # Run change detection pipeline
    try:
        shard_manager = ShardManager()
        detector = ChangeDetector(
            snapshot_store=SnapshotStore(),
            timeline_store=TimelineStore(),
            change_scorer=ChangeScorer(),
            embedder=OpenAIEmbedder(),
            shard_router=ShardRouter(shard_manager),
            idempotency=IdempotencyGuard(),
        )
        outcome = await detector.process_fetch(result, competitor_name)
        logger.info("Processed %s: %s", url, outcome.get("status", "unknown"))
    except Exception as exc:
        logger.error("Change detection failed for %s: %s", url, exc)
        raise


async def main() -> None:
    """Run the worker pool with the job handler."""
    from sentinel.ingestion.worker_pool import WorkerPool
    from sentinel.models.database import init_db

    logger.info("Initializing worker process...")

    # Initialize database
    await init_db()

    pool = WorkerPool(handler=handle_job)

    # Handle graceful shutdown
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(pool.stop()))
        except NotImplementedError:
            pass  # Windows doesn't support add_signal_handler

    await pool.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
