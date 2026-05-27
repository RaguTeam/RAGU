# AGENTS.md — RAGU Development Guide

## Table of Contents

- [Communication Language](#communication-language)
- [Project Overview](#project-overview)
- [Development Commands](#development-commands)
- [Workflow After Changes](#workflow-after-changes)
- [Code Conventions](#code-conventions)
- [Testing](#testing)
- [Key Invariants](#key-invariants)
- [Storage Contracts](#storage-contracts)
- [Prompt System](#prompt-system)
- [Extending RAGU](#extending-ragu)
- [Do Not Touch](#do-not-touch)
- [Code Quality](#code-quality)

---

## Communication Language

Reply to the user in the language they used (English, Russian, or any other). **Code comments must always be in English** regardless. Log messages and user-facing documentation match the user's language.

---

## Project Overview

RAGU (Retrieval-Augmented Graph Utility) is a modular GraphRAG engine for building, storing, and querying knowledge graphs from text. Entity and relation types follow the [NEREL](https://github.com/nerel-ds/NEREL) schema. Both English and Russian are supported via `Settings.language`.

Processing pipeline:

```
Documents -> Chunker -> List[Chunk]
  -> ArtifactExtractor -> List[Entity], List[Relation]
    -> EntitySummarizer / RelationSummarizer (merge + optional LLM summarization)
      -> Optional GraphBuilderModules (e.g., RemoveIsolatedNodes)
        -> Leiden community detection -> List[Community]
          -> CommunitySummarizer -> List[CommunitySummary]
            -> Index (persists graph + KV + vectors)

Query -> SearchEngine.a_search() -> retrieval context
      -> SearchEngine.a_query() -> LLM-generated answer
```

The full public API is re-exported from `ragu/__init__.py`. Top-level subpackages: `chunker`, `common`, `graph`, `models`, `search_engine`, `storage`, `triplet`, `utils`.

---

## Development Commands

The project uses **`uv`** for package management.

```bash
# Editable install
uv pip install -e .

# With test dependencies
uv pip install -e ".[test]"

# Full test suite (with coverage, per pytest.ini)
pytest tests/

# Fast tests only
pytest -m "not slow and not integration"

# Without coverage
pytest -q --no-cov
```

---

## Workflow After Changes

After modifying code, the agent **must**:

1. Run `pytest -m "not slow and not integration"` and ensure it passes.
2. If a new public class/function was added — re-export it from the subpackage `__init__.py` **and** from `ragu/__init__.py`.
3. If a new prompt was added — register it in `DEFAULT_PROMPT_TEMPLATES` (`ragu/common/prompts/prompt_storage.py`). There is no auto-discovery.
4. If a new dependency was added — update `pyproject.toml`.

---

## Code Conventions

### Imports
- **Absolute imports only**: `from ragu.common.logger import logger`. Relative imports (`from ..common import ...`) are forbidden.

### Naming
- **Classes**: PascalCase.
- **Functions/methods**: snake_case.
- **Async methods**: prefixed with `a_` (`a_search`, `a_embed_text`).
- **Sync wrappers** of async methods: no prefix, delegate via `always_get_an_event_loop()`.
- **Constants**: UPPER_SNAKE_CASE.
- **Private/internal**: single underscore prefix.
- **Files**: snake_case.

### Dataclasses
- Domain types: `@dataclass(slots=True)`.
- Immutable configs: `@dataclass(frozen=True, slots=True)`.
- Auto-ID pattern: `id: str = 'auto'` with `__post_init__` calling `compute_mdhash_id(...)`.

### Type Hints
- Required on all public methods.
- Python 3.10+ syntax: `X | None` (not `Optional[X]`), `list[str]` (not `List[str]`).
- Use `TypeVar` with bounds for generics: `NodeT = TypeVar("NodeT", bound=Node)`.
- Pydantic `BaseModel` for structured LLM output schemas.

### Async
- Core operations are `async def`.
- Batch operations use `tqdm_asyncio.gather` for concurrency with progress bars.

### Logging
- **loguru only**: `from ragu.common.logger import logger`. Never use stdlib `logging` directly (except the `LoguruAdapter` bridge for tenacity).

### Error Handling
- Raise **`ValueError`** for all validation errors. Do **not** introduce a custom exception hierarchy.
- Use `assert` for internal invariants.
- `tenacity` handles transient API errors in `CachedAsyncOpenAI`.

### Docstrings
- reStructuredText / Sphinx style (`:param`, `:return:`).

### Overrides
- Use `@override` from `typing_extensions` on subclass method implementations.

---

## Testing

- **pytest + pytest-asyncio**, `asyncio_mode = auto` — async tests need no decorator.
- Markers: `asyncio`, `slow`, `integration`.

### Critical: overriding `Settings`

`Settings` is a singleton. Direct assignment leaks across tests. **Always** use `monkeypatch`:

```python
monkeypatch.setattr(Settings, "storage_folder", str(tmp_path / "storage"))
```

### Useful fixtures and helpers
- `tests/kg_for_test/` — pre-built serialized knowledge graph for integration tests.
- `real_kg` fixture (`tests/search_engine/conftest.py`) — loads the pre-built graph.
- `DummyEmbedder` — returns constant vectors.
- `VDBBackendCase` (`tests/storage/conftest.py`) — parametrizes vector DB tests across NanoVDB, Qdrant-dense, Qdrant-BM42.
- `OpenAIMockServer` (`ragu/utils/testing/`) — deterministic mock LLM server.

---

## Key Invariants

These rules are not visible from class signatures but must always hold.

- **All LLM-driven modules inherit from `RaguGenerativeModule`** (`ragu/common/base.py`). It manages prompt loading and customization. Subclasses include `BaseArtifactExtractor`, `EntitySummarizer`, `RelationSummarizer`, `CommunitySummarizer`, `BaseEngine`.

- **Search engines inherit from `BaseEngine`** and must implement both `a_search` (retrieval) and `a_query` (LLM-generated answer). Sync wrappers (`search`, `query`) are provided by the base class.

- **Storage backends work with `Node` / `Edge` base classes, not with `Entity` / `Relation` directly.** This allows custom domain subclasses. Constructors must accept `node_cls` and `edge_cls` parameters and use `TypeVar` bounds, not hardcoded types.

- **Domain object IDs are deterministic MD5 hashes** computed by `compute_mdhash_id(content, prefix)`. This enables deduplication and incremental upserts.

- **`Settings.storage_folder` is the single source of truth** for persistence paths. Never hardcode paths.

---

## Storage Contracts

### Lifecycle hooks
All storage backends inherit from `BaseStorage` (`ragu/storage/base_storage.py`) and expose:
- `index_start_callback()` — call before writes.
- `index_done_callback()` — call after writes to flush/persist.
- `query_done_callback()` — call after queries.

Skipping these calls causes silent data loss.

### Upsert vs Update
- **Upsert** — merges with existing data via merge policies (e.g., `default_merge_entities_policy`); appends descriptions, merges chunk references.
- **Update** — replaces entirely.
- Both reject duplicate IDs within a single request.

### Edge validation
Edges cannot reference non-existent endpoints. `Index._validate_edge_endpoints_exist()` enforces this — do not bypass it.

### Default backends
- Graph: `NetworkXStorage` (GML, `nx.MultiDiGraph`).
- KV: `JsonKVStorage` (JSON).
- Vector: `NanoVectorDBStorage` (JSON, dense only).
- Production vector: `QdrantVectorDBStorage` (dense + sparse hybrid with RRF fusion).

---

## Prompt System

- Prompts are `RAGUInstruction` instances (frozen dataclass) registered in `DEFAULT_PROMPT_TEMPLATES` (`ragu/common/prompts/prompt_storage.py`). Refer to that file for the current list of names — do not duplicate it elsewhere.
- Each `RAGUInstruction` binds `messages: ChatMessages`, an optional `pydantic_model` for structured output, an optional `description`, and an optional `few_shot_formatter`.
- Rendering is done with **Jinja2** via `ChatMessages.render(**params)`.
- Few-shot examples are injected by `FewShotFormatter`; example selection is controlled by `ICLConfig` via four strategies: `"semantic"` (dense cosine similarity, requires `Embedder`), `"bm25"` (lexical matching via FastEmbed BM25, no embedder needed), `"hybrid"` (Reciprocal Rank Fusion of both), and `"random"` (uniform sampling baseline).

### Customization rule
**Never modify `DEFAULT_PROMPT_TEMPLATES` directly.** Use `module.update_prompt(name, instruction)` on the relevant `RaguGenerativeModule` instance.

---

## Extending RAGU

When adding a new component, keep these non-obvious rules in mind:

- **New search engines / extractors / builder modules** — inherit the relevant abstract base; check the abstract methods in source. The base class already wires up sync wrappers and prompt management.
- **Re-export new public types** from both the subpackage `__init__.py` and `ragu/__init__.py`.
- **New storage adapters** — use `TypeVar` bounds (`NodeT = TypeVar("NodeT", bound=Node)`), accept `node_cls` / `edge_cls` in the constructor, implement all three lifecycle callbacks. Never hardcode `Entity` / `Relation`.
- **New prompts** — define templates in `ragu/common/prompts/default_templates.py`, register a `RAGUInstruction` in `DEFAULT_PROMPT_TEMPLATES`, and (if structured output is needed) add a Pydantic model in `ragu/common/prompts/default_models.py`.

---

## Do Not Touch

- `tests/kg_for_test/` — snapshot of a pre-built knowledge graph used by integration tests. Regenerating it requires a separate, intentional procedure.
- `DEFAULT_PROMPT_TEMPLATES` contents — extend via the registration mechanism, do not edit existing entries to "fix" behavior for one call site (use `update_prompt`).

---

## Code Quality

This repository does **not** configure linters (ruff, flake8), type checkers (mypy), or formatters (black). Do not assume these tools are available and do not run them unless you explicitly add them to `pyproject.toml`.
