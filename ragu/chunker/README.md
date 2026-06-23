# Module: ragu.chunker

## Role in RAGU Pipeline

`ragu.chunker` is the first stage of the RAGU indexing pipeline. It transforms raw documents into `Chunk` objects that can be sent to artifact extraction, vector storage, and naive retrieval.

Pipeline position:

```text
documents -> chunker -> List[Chunk] -> extraction / graph builder / chunk vector index
```

## Overview

The module exists to give the rest of RAGU stable text units with deterministic IDs. Every chunk stores its text, order inside the source document, source document ID, and optional token count. `KnowledgeGraph.build_from_docs()` uses the configured chunker before running extraction. If no chunker is configured, each input document is treated as one chunk.

## Key Components

### Chunk

Dataclass from `ragu.chunker.types`.

- Purpose: immutable-ish text unit passed between indexing and retrieval stages.
- Important fields: `content`, `chunk_order_idx`, `doc_id`, `num_tokens`.
- ID behavior: `id` is generated in `__post_init__` from `content` with prefix `chunk-`.

### BaseChunker

Abstract interface for chunkers.

- Purpose: standardizes `split(documents) -> list[Chunk]`.
- Used by: `KnowledgeGraph` and `InMemoryGraphBuilder`.

### SimpleChunker

Sentence-aware fixed-size chunker.

- Purpose: split text by `razdel.sentenize`, then merge sentences until `max_chunk_size` characters.
- Important parameters: `max_chunk_size`, `overlap`.

```python
import asyncio
from ragu.chunker.types import Chunk
from ragu.chunker import SimpleChunker

documents = [
    "First document",
    "Second document"
]

async def main():
    chunker = SimpleChunker(max_chunk_size=512, overlap=0)
    chunks: list[Chunk] = chunker(documents)

    print(chunks)

asyncio.run(main())
```

### SemanticTextChunker

Sentence-transformer based semantic splitter.

- Purpose: split by sentence boundaries and recursively separate less similar sentence spans.
- Important parameters: `model_name`, `max_chunk_size`, `device`.
- External dependency: `sentence_transformers`.

```python
import asyncio
from ragu.chunker.types import Chunk
from ragu.chunker import SemanticTextChunker

documents = [
    "First document",
    "Second document"
]

async def main():
    chunker = SemanticTextChunker(
        model_name="",
        max_chunk_size=1024,
        device="cuda:0"
    )
    chunks: list[Chunk] = chunker(documents)

    print(chunks)

asyncio.run(main())
```

### SmartSemanticChunker

Wrapper around `smart_chunker.SmartChunker`.

- Purpose: use a reranker-based algorithm for long-document chunking.
- Important parameters: `reranker_name`, `max_chunk_length`, `minibatch_size`, `device`.
- External dependency: `smart_chunker`.

```python
import asyncio
from ragu.chunker.types import Chunk
from ragu.chunker import SmartSemanticChunker

documents = [
    "First document",
    "Second document"
]

async def main():
    chunker = SmartSemanticChunker(
        max_chunk_length=1024,
        minibatch_size=16
        # And more parameter
    )
    chunks: list[Chunk] = chunker(documents)

    print(chunks)

asyncio.run(main())
```

## Data Flow

Input: `str` or `list[str]` documents.

Output: `list[Chunk]`.

Used by:

- `ragu.graph.KnowledgeGraph.build_from_docs`
- `ragu.graph.InMemoryGraphBuilder.extract_graph`
- `ragu.triplet` extractors
- `ragu.graph.Index.upsert_chunks`
- `ragu.search_engine.NaiveSearchEngine`

## Usage Examples

### Example 1 - Minimal usage

```python
from ragu.chunker import SimpleChunker

chunker = SimpleChunker(max_chunk_size=120, overlap=20)
chunks = chunker.split("Python was created by Guido van Rossum. It is widely used.")

for chunk in chunks:
    print(chunk.id, chunk.doc_id, chunk.chunk_order_idx, chunk.content)
```

### Example 2 - Pipeline usage

```python
import asyncio

from ragu import BuilderArguments, KnowledgeGraph, SimpleChunker
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.openai import CachedAsyncOpenAI


async def main():
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
        chunker=SimpleChunker(max_chunk_size=80),
        builder_settings=BuilderArguments(build_only_vector_context=True),
    )

    await graph.build_from_docs([
        "RAGU builds knowledge graphs from text.",
        "Naive search uses chunk vectors.",
    ])

    chunk_ids = await graph.index.chunks_kv_storage.all_keys()
    chunks = await graph.get_chunks(chunk_ids)
    print([chunk.content for chunk in chunks if chunk])


asyncio.run(main())
```

## Integration Points

- Extraction: `BaseArtifactExtractor.extract()` receives `Chunk` instances and writes `source_chunk_id` into entities and relations.
- Storage: `Index.upsert_chunks()` stores chunk metadata in KV storage and dense/sparse embeddings in vector storage.
- Retrieval: `GraphRetriever.query_chunks()` resolves vector hits back into `Chunk` objects for `NaiveSearchEngine`.
- Settings: chunkers do not use global storage settings directly; `KnowledgeGraph` controls where chunk outputs are persisted.

## Configuration

- `SimpleChunker(max_chunk_size, overlap=0)`: character budget and optional character overlap.
- `SemanticTextChunker(model_name, max_chunk_size, device="cuda:0")`: sentence-transformer model and token budget.
- `SmartSemanticChunker(...)`: reranker model, tokenizer callables, device, `max_chunk_length`, and batching.

## Dependencies

Internal:

- `ragu.chunker.types.Chunk`
- `ragu.utils.ragu_utils.compute_mdhash_id`

External:

- `razdel`
- `tqdm`
- `nltk`
- `numpy`
- optional `sentence_transformers`
- optional `smart_chunker`

## Notes / Pitfalls

- `Chunk.id` is content-derived. Duplicate chunk text across documents produces the same ID and is deduplicated by `KnowledgeGraph.build_from_docs()`.
- `doc_id` is currently derived from the whole document hash inside chunkers.
- `SimpleChunker.max_chunk_size` is measured in characters, not tokens.
- Semantic chunkers load local ML models and may require GPU-compatible configuration.
