# Development Commands

Install editable package:
```bash
uv pip install -e .
```

Install with test dependencies:
```bash
uv pip install -e ".[test]"
```

Run all tests:
```bash
pytest tests/
```

Run fast tests only (skip slow and integration):
```bash
pytest -m "not slow and not integration"
```

Run tests with coverage (without coverage report to save time):
```bash
pytest -q --no-cov
```

# Testing

Tests are async-first. All async tests must be marked with `@pytest.mark.asyncio`. Pytest is configured with `asyncio_mode = auto`.

Test markers:
- `asyncio`: async tests (default deselected with `-m "not asyncio"`)
- `slow`: slow tests (deselect with `-m "not slow"`)
- `integration`: integration tests, e.g., Memgraph (deselect with `-m "not integration"`)

The test suite uses a pre-built knowledge graph fixture in `tests/kg_for_test/` for integration-level tests.

# Global State Management

`Settings` is a singleton that controls storage location and language. Tests that need temporary storage MUST use:
```python
monkeypatch.setattr(Settings, "storage_folder", str(tmp_path / "storage"))
```

Do not directly assign to `Settings.storage_folder` in tests without monkeypatch, as changes persist across tests.

# Entry Points

`KnowledgeGraph` is the main facade. Initialize with:
- `llm`: LLM instance (or None if only building vectors)
- `embedder`: Embedder instance
- `chunker`: Chunker instance
- `artifact_extractor`: Extractor for entities/relations (e.g., `TwoStageArtifactsExtractorLLM`)
- `builder_settings`: `BuilderArguments` for pipeline config
- `storage_settings`: `StorageArguments` for backend selection

All operations are async: use `await knowledge_graph.build_from_docs(docs)` and `await search_engine.a_query(...)`.

# Storage Contracts

Storage adapters operate on `Node` and `Edge` protocols, NOT on `Entity` and `Relation` directly. This allows custom subclasses of `Entity`/`Relation`. When implementing or using storage backends, pass the concrete node/edge classes to the storage constructor.

Default storage backends:
- Graph: `NetworkXStorage` (GML file)
- KV: `JsonKVStorage`
- Vector: `NanoVectorDBStorage` (JSON file)

For production, use `QdrantVectorDBStorage` with `sparse_type` for hybrid retrieval (BM25, BM42, SPLADE).

# Code Quality

This repo does NOT configure linting (ruff, flake8), typechecking (mypy), or formatting (black). Do not assume these tools are available unless you add them.
