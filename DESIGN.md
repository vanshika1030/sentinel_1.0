# Sentinel — Architecture Decision Records

Short rationale for the key technology choices in Sentinel.

---

## 1. Why Redis Streams over Kafka

**Decision:** Use Redis Streams for the crawl job queue.

**Context:** We need a message queue for distributing crawl jobs to workers with at-least-once delivery, consumer groups, and dead-letter queue support.

**Rationale:**
- Redis Streams provide consumer groups (`XREADGROUP`), message acknowledgment (`XACK`), pending message recovery (`XAUTOCLAIM`), and dead-letter routing — all the semantics we need.
- Redis is already required for the idempotency guard (content hash deduplication), so adding Streams incurs zero additional infrastructure.
- Kafka is designed for massive throughput (millions of events/sec) and multi-datacenter replication. Our workload is ~10-100 crawl jobs/hour — three orders of magnitude below Kafka's sweet spot.
- Kafka requires ZooKeeper or KRaft, a JVM runtime, and significant configuration overhead. Redis is a single binary with sub-millisecond latency.
- For an MVP, operational simplicity wins. If we later need replay, compaction, or cross-region replication, Kafka is a straightforward upgrade.

---

## 2. Why Qdrant over Pinecone/Weaviate

**Decision:** Use Qdrant as the vector database.

**Context:** We need a vector store that supports self-hosted deployment, multiple instances for sharding, and efficient ANN search.

**Rationale:**
- **Self-hosted:** Qdrant runs as a single Docker container with persistent storage. No cloud vendor lock-in, no API keys, no usage-based billing.
- **Sharding control:** By running multiple Qdrant instances and implementing our own consistent hash ring, we have full control over data placement and can add shards without re-indexing everything.
- **Performance:** Qdrant uses HNSW with quantization, delivering sub-10ms search latency for collections under 1M vectors — well within our scale.
- **API simplicity:** The Python client provides synchronous and async interfaces with a clean, typed API.
- Pinecone is fully managed (no self-hosting option for air-gapped or cost-sensitive deployments). Weaviate is excellent but heavier to operate and its module system adds complexity we don't need.

---

## 3. Why Consistent Hashing over Modulo

**Decision:** Use a consistent hash ring with virtual nodes for shard routing.

**Context:** Documents need to be distributed across Qdrant shards, and we want to support adding/removing shards without full re-indexing.

**Rationale:**
- **Modulo hashing** (`hash(key) % N`) is simple but catastrophic when N changes: adding a shard remaps ~100% of keys, requiring a full data migration.
- **Consistent hashing** remaps only `~1/N` of keys when a node is added or removed, enabling live shard scaling.
- **Virtual nodes** (150 per real node) solve the uneven distribution problem that basic consistent hashing suffers from, achieving near-uniform key distribution.
- The implementation is ~100 lines of Python using only `hashlib` and `bisect` — no external dependencies, fully testable, easy to reason about.

---

## 4. Why Hybrid Search over Pure Vector Search

**Decision:** Combine vector search with BM25, Reciprocal Rank Fusion, and cross-encoder re-ranking.

**Context:** Users ask varied questions — some are semantic ("What's their strategy?"), some are keyword-heavy ("pricing tier changes"), some mix both.

**Rationale:**
- **Pure vector search** excels at semantic similarity but struggles with exact keyword matches. A user asking about "pricing" may miss documents that use the word "pricing" but are semantically distant from the query embedding.
- **BM25** handles exact term matching well but misses paraphrases and semantic relationships.
- **Reciprocal Rank Fusion (RRF)** merges both ranked lists without requiring score normalization — it's parameter-free (aside from the k constant) and robust.
- **Cross-encoder re-ranking** provides a final quality boost by jointly encoding the query and each candidate, catching relevance signals that bi-encoder embeddings miss.
- The pipeline is: Vector (20 candidates) + BM25 (20 candidates) → RRF merge → Cross-encoder → Top 5. This balances quality and latency effectively.

---

## 5. Why Circuit Breaker Pattern

**Decision:** Implement circuit breakers around all external service calls.

**Context:** Sentinel depends on Redis, PostgreSQL, Qdrant, and external websites. Any of these can become temporarily unavailable.

**Rationale:**
- Without circuit breakers, a downed Qdrant shard causes every query to block for the full timeout (3s), wasting resources and degrading UX.
- The circuit breaker **fails fast** in OPEN state, returning errors immediately instead of waiting.
- The HALF-OPEN state allows automatic recovery detection without manual intervention.
- Combined with scatter-gather's tolerance for partial results, a single shard outage degrades results but doesn't block the system.
