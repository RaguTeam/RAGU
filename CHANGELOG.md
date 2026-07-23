# Changelog

## [0.0.4] - 2026-07-24

### Added
- Neo4j graph storage backend (`Neo4jStorage`) as an alternative to the default
  NetworkX file backend: server-backed graphs, concurrent access, and Cypher.
- Streaming search: `stream_query` on the search engines and
  `stream_chat_completion` on the LLM clients, for token-by-token answers.
- Batched vector-DB queries and batched search, so multiple queries are scored
  in a single pass instead of one at a time.
- Configurable distance metric (`cosine` / `dot`) for the built-in dense
  vector store, with the metric persisted and validated on load.
- `close()` on storage backends and on `Index`, to release connection pools of
  server-backed backends (Neo4j, remote Qdrant).

### Changed
- **Breaking:** `BaseGraphStorage.get_edges` (and `Index.get_edges`,
  `KnowledgeGraph.get_relations`) now return one edge list per spec
  (`List[List[Edge]]`) instead of a flat `List[Optional[Edge]]`. RAGU graphs are
  multigraphs, so a node pair may hold several edges; the list-per-spec shape
  keeps results aligned with the input specs and no longer drops edges.
- **Breaking:** graph-building failures are no longer swallowed. A failing
  extraction, summarization, module, or clustering step now propagates out of
  `build_from_docs` instead of logging a warning and continuing with a partial
  graph.
- **Breaking:** `GraphBuilderModule` is now abstract; empty ids on `Entity` /
  `Relation` / `Community` / `CommunitySummary` are rejected.
- The `neo4j` driver moved from an optional extra to a regular dependency.
- Qdrant upserts use `models.Batch`, cutting insert time roughly threefold on
  large batches.
- Graph nodes and edges declare their grouping field (`label_field`), so the
  Neo4j backend derives labels, relationship types, and indexes from the types
  rather than guessing.

### Fixed
- Silently misaligned vectors from `get_points_by_ids` in the dense vector store
  (positions were read from a filtered subset instead of the full matrix).
- Storage writes are now atomic (temp file + `os.replace`), so an interrupted
  save can no longer truncate a KV store or vector index.
- Dropped community summaries reappeared after an empty reclustering; the empty
  state is now persisted.
- The embedding-dimension check rejected consistent configurations because it
  counted values instead of comparing them.

## [0.0.3] - 2026-06-05

- Added in-context learning and few-shot support for artifact extractors.
- Added built-in examples and selection strategies for ICL: semantic, BM25, hybrid, and random.
- Improved OpenAI-compatible server and embedder error handling.
- Added embedder input auto-truncation and `GlobalSettings` serialization.
- Fixed query embedding typing in `GraphRetriever` and an awaited coroutine bug in `RaguLmArtifactExtractor`.

## [0.0.2] - 2026-04-27

- Added Qdrant vector storage support.
- Added sparse embeddings and hybrid retrieval.
- Added `GraphRetriever`, retrieval metrics, and index consistency checks.
- Search methods now return relevance scores.
- Moved clustering from `graspologic` to `graspologic-native`.

## [0.0.1] - 2026-03-21

- Started the new `0.0.x` version line.
- Reworked the graph builder, index, storage, and model interfaces.
- Added naive, local, global, and mix search engines with query planning.
- Added CRUD operations for graph, KV, and vector storage.
- Added RAGU-lm prompt support, the two-stage extractor, caching, tests, and a prebuilt knowledge graph fixture.
