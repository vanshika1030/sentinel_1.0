# Sentinel

**Distributed Real-Time Competitive Intelligence Engine**

Monitor competitors' websites, docs, changelogs, and GitHub repos — detect changes, index everything into a sharded vector database, and ask natural language questions about competitive movements.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SENTINEL ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────┐    ┌──────────────┐    ┌─────────────────────────┐    │
│  │  User   │───▶│  FastAPI      │───▶│  RAG Engine             │    │
│  │  Query  │    │  /api/query   │    │  Embed → Search → LLM  │    │
│  └─────────┘    └──────────────┘    └────────────┬────────────┘    │
│                                                   │                 │
│                              ┌────────────────────┴───────────┐    │
│                              │     Scatter-Gather Query        │    │
│                              │  ┌──────────┐  ┌──────────┐    │    │
│                              │  │ Qdrant-0 │  │ Qdrant-1 │    │    │
│                              │  │ (shard)  │  │ (shard)  │    │    │
│                              │  └──────────┘  └──────────┘    │    │
│                              │   Consistent Hash Ring          │    │
│                              └────────────────────────────────┘    │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   Ingestion Pipeline                         │   │
│  │  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐  │   │
│  │  │Scheduler │───▶│ Redis Stream │───▶│ Consumer Workers  │  │   │
│  │  │(cron)    │    │ (crawl_jobs) │    │ (async pool)      │  │   │
│  │  └──────────┘    └──────────────┘    └────────┬─────────┘  │   │
│  │                                                │            │   │
│  │  ┌────────────────────────────────────────────┘            │   │
│  │  │  Sources: Website │ GitHub │ RSS                         │   │
│  │  │  Rate Limiter │ Idempotency Guard │ Retry w/ Backoff     │   │
│  │  └─────────────────────────────────────────────────────────│   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │               Change Detection Pipeline                      │   │
│  │  Snapshot → Diff → Score Significance → Store Timeline       │   │
│  │  PostgreSQL: snapshots + changes tables                      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Resilience: Circuit Breaker │ Retry + Backoff │ Health      │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Features

- **Multi-source ingestion** — crawl websites, GitHub repos, and RSS feeds
- **Sharded vector search** — consistent hash ring across 2 Qdrant instances
- **Hybrid retrieval** — vector search + BM25 + Reciprocal Rank Fusion + cross-encoder re-ranking
- **Change detection** — diff-based with significance scoring (semantic + keyword + structural)
- **Natural language Q&A** — RAG pipeline with source citations
- **Fault tolerance** — circuit breakers, retry with exponential backoff, health checks
- **Message queue** — Redis Streams with consumer groups, DLQ, and idempotency
- **Scheduled crawling** — configurable intervals with on-demand trigger

---

## Tech Stack

| Component       | Technology                           |
|-----------------|--------------------------------------|
| Language        | Python 3.11+                         |
| API Framework   | FastAPI (async)                      |
| RAG Framework   | LangChain                            |
| Vector Database | Qdrant (2 shards via Docker)         |
| Message Queue   | Redis Streams                        |
| Storage         | PostgreSQL 16                        |
| Embeddings      | OpenAI `text-embedding-3-small`      |
| LLM             | OpenAI `gpt-4o-mini`                 |
| HTTP Client     | httpx (async)                        |
| HTML Parsing    | BeautifulSoup4 + lxml                |
| Containers      | Docker + Docker Compose              |
| Search Ranking  | rank-bm25 + sentence-transformers    |

---

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/vanshika1030/sentinel_1.0.git
cd sentinel_1.0
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-your-key-here
```

### 2. Start everything

```bash
docker-compose up -d --build
```

This starts 6 services:
- `sentinel-api` — FastAPI on port 8000
- `sentinel-worker` — 2 async ingestion workers
- `redis` — Redis 7 on port 6379
- `postgres` — PostgreSQL 16 on port 5432
- `qdrant-0` — Qdrant shard on port 6333
- `qdrant-1` — Qdrant shard on port 6334

### 3. Verify

```bash
curl http://localhost:8000/api/health
```

---

## API Reference

### Query Intelligence

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What features did Vercel ship recently?"}'
```

```bash
# Filter by competitor and time range
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Show me pricing changes",
    "competitor": "supabase",
    "since": "2025-01-01T00:00:00Z"
  }'
```

### Manage Competitors

```bash
# List all competitors
curl http://localhost:8000/api/competitors

# Add a competitor
curl -X POST http://localhost:8000/api/competitors \
  -H "Content-Type: application/json" \
  -d '{
    "name": "stripe",
    "urls": [
      {"url": "https://stripe.com/blog", "source_type": "website"},
      {"url": "https://github.com/stripe/stripe-node", "source_type": "github"}
    ]
  }'

# Get change timeline
curl "http://localhost:8000/api/competitors/vercel/changes?significant_only=true"
```

### Ingest Control

```bash
# Trigger a crawl for all competitors
curl -X POST http://localhost:8000/api/ingest/trigger \
  -H "Content-Type: application/json" \
  -d '{}'

# Trigger for a specific competitor
curl -X POST http://localhost:8000/api/ingest/trigger \
  -H "Content-Type: application/json" \
  -d '{"competitor_name": "vercel"}'

# Check ingestion status
curl http://localhost:8000/api/ingest/status
```

### Health Check

```bash
curl http://localhost:8000/api/health
# {"status":"healthy","components":{"redis":"up","qdrant_0":"up","qdrant_1":"up","postgres":"up"}}
```

---

## Architecture Deep-Dive

### Sharded Vector Search

Sentinel distributes documents across multiple Qdrant instances using a **consistent hash ring** implemented from scratch. Each document ID is hashed via SHA-256 and mapped to a shard through 150 virtual nodes per real node, ensuring even distribution. Queries use a **scatter-gather** pattern: the same search is sent to all healthy shards concurrently, results are merged by score, deduplicated, and the global top-K is returned. If a shard times out (default: 3s), the system continues with partial results rather than failing entirely.

### Change Detection

Every crawl result goes through a multi-stage change detection pipeline. The system stores content snapshots in PostgreSQL, computes unified diffs against the previous version, and scores significance using three weighted signals: **structural diff ratio** (how much text changed — 30%), **semantic distance** (cosine distance between old and new embeddings — 40%), and **keyword triggers** (matches against terms like "pricing", "launch", "breaking" — 30%). Changes scoring above 0.5 are flagged as significant, and their diff text is embedded into the vector DB so users can directly search for "what changed."

### Fault Tolerance

The system implements a proper **circuit breaker state machine** (CLOSED → OPEN → HALF-OPEN) with sliding window failure tracking, configurable thresholds, and automatic recovery. This wraps every external service call. Combined with **exponential backoff with jitter** for retries, **per-domain rate limiting** via token buckets, and a **dead-letter queue** for messages that fail after 3 attempts, Sentinel degrades gracefully when downstream services are unavailable.

---

## Development

```bash
# Run tests
make test

# Lint
make lint

# View logs
make logs

# Rebuild
make rebuild

# Clean everything (including volumes)
make clean
```

---

## License

MIT
