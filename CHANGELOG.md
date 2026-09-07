# Changelog

## [0.1.0] - 2026-09-08

### Added
- Ontology layer (`ragu.triplet.ontology`): the entity and relation vocabulary
  and the rules that constrain it in one object — `domain` / `range` per
  predicate, an entity type hierarchy, symmetric and inverse predicates,
  `aliases` and `inverse_aliases`, and `retype_when` rules that rewrite a
  predicate for specific endpoint types. Three ways to build one:
  - `Ontology.builtin("nerel")` — a named built-in;
  - `Ontology.from_yaml(path)` — a YAML document, optionally `extends:`
    another ontology;
  - `Ontology.from_type_lists(entities, relations)` — vocabulary only, no rules.
- `OntologyValidator`, applied by both LLM extractors between the model output
  and `Entity` / `Relation` construction. Checks types outside the vocabulary,
  `domain` / `range` violations, self-loops, and endpoints missing from the
  chunk, and returns a `ValidationReport` with a per-check breakdown.
- `ValidationPolicies` chooses what happens per check: `COERCE` (repair through
  an alias, a fuzzy match, a `retype_when` rule, or an endpoint swap — and only
  if the result satisfies the ontology), `DROP`, `KEEP` (count only), or
  `RAISE`. Presets: `permissive()`, `strict()`, `failing()`.
- `ragu/triplet/ontology/builtin/nerel.yaml`: NEREL as an ontology — 29 entity
  types, 49 predicates, with domain/range rules. It is now the single source of
  truth for `NEREL_ENTITY_TYPES` / `NEREL_RELATION_TYPES`, which stay available
  in their historical inline format.
- `show_type_signatures` on the LLM extractors renders each predicate in the
  prompt with its signature, `WORKPLACE [PERSON -> ORGANIZATION|FACILITY]`, and
  adds a legend for the notation. Both go into the system message, so a backend
  with prefix caching prefills them once for the whole batch. Off by default.
- `prune_relation_types` on `TwoStageArtifactsExtractorLLM` offers only the
  predicates admissible between the entity types found in the chunk.
- `min_cluster_size` in `BuilderArguments` drops communities below that many
  entities right after community detection, so they are neither summarized nor
  stored. `GlobalSearchParams(min_cluster_size=...)` applies the same threshold
  at query time, over communities already in the index.
- Precise types on the public API: a concrete `output_schema` carries through to
  the return type of `LLM.chat_completion` / `batch_chat_completion` instead of
  widening to `BaseModel | str`, and `BaseEngine[ParamsT, RetrieveT]` types each
  engine's own parameters and retrieval.

### Changed
- **Breaking:** the LLM extractors take a single `ontology` argument instead of
  `entity_types` / `relation_types`. That object supplies both the type list
  injected into the prompt and the rules enforced on what comes back.
  `ontology=None` restores the previous behaviour: the model invents its own
  labels and nothing is checked.

  ```diff
  - extractor = ArtifactsExtractorLLM(llm, entity_types=[...], relation_types=[...])
  + extractor = ArtifactsExtractorLLM(llm, ontology=Ontology.builtin("nerel"))
  ```

- **Breaking:** `BaseEngine.__init__` takes `prompts` as a regular argument and
  everything after it keyword-only; the `*args` / `**kwargs` passthrough is gone.

  ```diff
  - super().__init__(llm=llm, prompts=_PROMPTS, *args, **kwargs)
  + super().__init__(llm, prompts=_PROMPTS, max_context_length=max_context_length)
  ```

- **Breaking:** `Index` is no longer generic. The storage backends stay
  parameterized and materialize records through `node_cls` / `edge_cls`, so a
  subclass of `Entity` still round-trips.

  ```diff
  - index: Index[Entity, Relation] = Index(...)
  + index: Index = Index(...)
  ```

- **Breaking:** `BaseArtifactExtractor.extract` takes `List[Chunk]` instead of
  `Iterable[Chunk]`.
- **Breaking:** `upsert_nodes` / `upsert_edges` on the graph storage backends
  take a `List` instead of an `Iterable`.
- `BaseGraphStorage` defines an `__init__` that records the node and edge
  classes a backend materializes, so a custom backend can call it instead of
  assigning `_node_cls` / `_edge_cls` itself. Not required: the built-in
  backends still assign them directly.
- Entity names and relation endpoints are stripped of surrounding whitespace as
  they are parsed, so `" Пушкин"` and `"Пушкин"` no longer become two entities.
- Development: the package is type-checked under a `mypy.ini` with
  `disallow_untyped_defs` and `check_untyped_defs`.

### Deprecated
- `RaguLmArtifactExtractor`, in favour of `TwoStageArtifactsExtractorLLM` and
  `ArtifactsExtractorLLM`. Constructing it emits a `DeprecationWarning`; it will
  be removed in a future release. `@deprecated` now works on a class without
  turning it into a function.

### Fixed
- Two-stage extraction discarded the entities of a chunk whenever its relation
  stage produced nothing — and every entity of the batch when the relation call
  failed outright. The entities are now kept and returned without relations.
- The artifact-validation prompt never received the list of allowed relation
  types, so the validation pass could rewrite a relation into a type outside the
  vocabulary.

## [0.0.5] - 2026-08-13

### Changed

- Updated dependency constraints with upper version bounds to prevent incompatible package upgrades.

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
