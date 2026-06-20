"""Database initialization script.

Creates all tables defined in the SQLAlchemy models.
Run this before first use or after model changes.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinel.models.database import init_db, close_db


async def main() -> None:
    """Initialize the database schema."""
    print("Initializing Sentinel database...")
    try:
        await init_db()
        print("✓ Database tables created successfully")
    except Exception as exc:
        print(f"✗ Failed to initialize database: {exc}")
        raise
    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(main())
