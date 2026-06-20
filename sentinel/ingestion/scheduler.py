"""Crawl scheduler — periodically re-crawls all configured competitors.

Runs as a background asyncio task, loading competitors from YAML
and pushing crawl jobs at a configurable interval.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import yaml

from sentinel.core.config import settings
from sentinel.ingestion.producer import JobProducer

logger = logging.getLogger(__name__)


class CrawlScheduler:
    """Schedules periodic crawl jobs for all configured competitors."""

    def __init__(self, producer: JobProducer | None = None) -> None:
        """Initialize the scheduler.

        Args:
            producer: Job producer for pushing crawl jobs.
        """
        self._producer = producer or JobProducer()
        self._running = False
        self._interval = settings.crawl_interval_hours * 3600

    def _load_competitors(self) -> list[dict[str, Any]]:
        """Load competitor configurations from YAML file.

        Returns:
            List of competitor configuration dicts.
        """
        config_path = Path(settings.competitors_file)
        if not config_path.exists():
            logger.warning("Competitors file not found: %s", config_path)
            return []

        with open(config_path) as f:
            data = yaml.safe_load(f)

        return data.get("competitors", []) if data else []

    async def _crawl_all(self) -> None:
        """Push crawl jobs for all configured competitor URLs."""
        competitors = self._load_competitors()
        jobs = []
        for competitor in competitors:
            name = competitor["name"]
            for url_config in competitor.get("urls", []):
                jobs.append({
                    "url": url_config["url"],
                    "source_type": url_config["source_type"],
                    "competitor_name": name,
                })

        if jobs:
            await self._producer.push_batch(jobs)
            logger.info("Scheduled %d crawl jobs for %d competitors", len(jobs), len(competitors))
        else:
            logger.warning("No competitors configured for crawling")

    async def _crawl_competitor(self, competitor_name: str) -> None:
        """Push crawl jobs for a specific competitor.

        Args:
            competitor_name: Name of the competitor to crawl.
        """
        competitors = self._load_competitors()
        jobs = []
        for competitor in competitors:
            if competitor["name"] == competitor_name:
                for url_config in competitor.get("urls", []):
                    jobs.append({
                        "url": url_config["url"],
                        "source_type": url_config["source_type"],
                        "competitor_name": competitor_name,
                    })
                break

        if jobs:
            await self._producer.push_batch(jobs)
            logger.info("Scheduled %d crawl jobs for %s", len(jobs), competitor_name)

    async def run(self) -> None:
        """Run the scheduler loop — initial crawl + periodic re-crawl."""
        self._running = True
        logger.info("Scheduler started (interval: %.1f hours)", settings.crawl_interval_hours)

        # Initial crawl on startup
        await self._crawl_all()

        while self._running:
            try:
                await asyncio.sleep(self._interval)
                await self._crawl_all()
            except asyncio.CancelledError:
                logger.info("Scheduler cancelled")
                break
            except Exception as exc:
                logger.error("Scheduler error: %s", exc)
                await asyncio.sleep(60)  # Wait before retry

    def stop(self) -> None:
        """Signal the scheduler to stop."""
        self._running = False

    async def trigger_crawl(self, competitor_name: str | None = None) -> None:
        """Manually trigger a crawl.

        Args:
            competitor_name: Specific competitor to crawl, or None for all.
        """
        if competitor_name:
            await self._crawl_competitor(competitor_name)
        else:
            await self._crawl_all()
