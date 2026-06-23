# Module: ragu.storage.kv_storage_adapters

## Role in RAGU Pipeline

This package provides key-value storage for chunks, community metadata, and community summaries.

Pipeline position:

```text
Chunk/Community/CommunitySummary -> KV storage -> retrieval context resolution
```

## Overview

KV adapters store JSON-like payloads that do not belong in the graph backend or vector database. The default implementation is local and file-backed.

## Key Components

### JsonKVStorage

- Purpose: persistent dictionary stored as a JSON file.
- Important methods: `all_keys`, `get_by_id`, `get_by_ids`, `filter_keys`, `upsert`, `delete`, `drop`.
- Important parameters: `storage_folder`, `filename`.

## Data Flow

Input: mappings from IDs to chunk/community/summary payloads.

Output: ordered lookup results with `None` for missing IDs.

Used by:

- `Index.chunks_kv_storage`
- `Index.community_kv_storage`
- `Index.community_summary_kv_storage`
- `GlobalSearchEngine`

## Usage Examples

### Example 1 - Minimal usage

```python
import asyncio
import tempfile

from ragu.storage.kv_storage_adapters.json_storage import JsonKVStorage


async def main():
    with tempfile.TemporaryDirectory() as directory:
        storage = JsonKVStorage(storage_folder=directory, filename="chunks.json")
        await storage.upsert({"chunk-1": {"content": "RAGU", "doc_id": "doc-1"}})
        await storage.index_done_callback()
        print(await storage.get_by_id("chunk-1"))


asyncio.run(main())
```

### Example 2 - Pipeline usage

```python
from ragu import BuilderArguments, KnowledgeGraph, StorageArguments
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.openai import CachedAsyncOpenAI
from ragu.storage.kv_storage_adapters.json_storage import JsonKVStorage


client = CachedAsyncOpenAI(
    base_url="https://api.openai.com/v1",
    api_key="dummy-api-token",
)
embedder = EmbedderOpenAI(
    client=client,
    model_name="text-embedding-3-small",
    dim=1536,
)

graph = KnowledgeGraph(
    llm=None,
    embedder=embedder,
    builder_settings=BuilderArguments(build_only_vector_context=True),
    storage_settings=StorageArguments(kv_storage_type=JsonKVStorage),
)
```

## Integration Points

- Chunk KV records are resolved after vector chunk search.
- Community summaries are read by `GlobalSearchEngine`.
- Community metadata links cluster IDs to entity and relation IDs.

## Configuration

`StorageArguments` provides separate kwargs for chunks, communities, and summaries:

- `chunks_kv_storage_kwargs`
- `communities_kv_storage_kwargs`
- `summary_kv_storage_kwargs`

## Dependencies

- Python `json`

## Notes / Pitfalls

- `upsert()` updates memory; `index_done_callback()` persists to disk.
- `drop()` clears in-memory data and should be followed by `index_done_callback()` when persistence matters.
