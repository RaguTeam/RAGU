# Module: ragu.graph

## Role in RAGU Pipeline

`ragu.graph` is the central indexing layer. It receives chunks and extracted artifacts, builds a knowledge graph, clusters it into communities, stores graph/vector/KV state, and exposes retrieval helpers.

Pipeline position:

```text
Chunk -> Entity/Relation extraction -> graph build/summarize/cluster -> Index
Index -> GraphRetriever -> search engines
```

## Overview

The module exists to keep graph construction and graph persistence independent from the model and storage implementations. `KnowledgeGraph` is the high-level facade. `InMemoryGraphBuilder` runs extraction, summarization, optional post-processing modules, and Leiden community detection. `Index` coordinates graph storage, vector storage, and JSON-like KV storage.

## Key Components

### KnowledgeGraph

High-level facade for build, CRUD, reindexing, and storage access.

- Purpose: orchestrates chunking, extraction, graph construction, vectorization, and persistence.
- Important methods: `build_from_docs`, `upsert_entities`, `upsert_relations`, `get_entities`, `get_relations`, `get_chunks`, `a_read_lock`, `reindex_community`, `reindex_descriptions`, `reindex_graph`.
- Important parameters: `llm`, `embedder`, optional `sparse_embedder`, `chunker`, `artifact_extractor`, `builder_settings`, `storage_settings`.

```python
from ragu.graph.knowledge_graph import KnowledgeGraph
from ragu.graph.graph_builder_pipeline import BuilderArguments
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.openai import CachedAsyncOpenAI

client = CachedAsyncOpenAI(
    base_url="https://api.openai.com/v1",
    api_key="dummy-api-token"
)
embedder = EmbedderOpenAI(
    client=client,
    model_name="text-embedding-3-small",
    dim=1536
)

graph = KnowledgeGraph(
    llm=None,
    embedder=embedder,
    builder_settings=BuilderArguments(build_only_vector_context=True),
)
print(graph.language)
```

### Entity

Graph node dataclass.

- Purpose: semantic node extracted from source chunks.
- Important fields: `entity_name`, `entity_type`, `description`, `source_chunk_id`, `documents_id`, `clusters`, `id`.
- ID behavior: defaults to `ent-...` hash of name and type.
- Extension point: `Entity` is the base graph model for RAGU entities and can be inherited to add domain-specific fields or methods. Custom entity classes should preserve the `Node` contract used by storage adapters.

> NOTE: custom entity extraction is not supported in RAGU versions 0.0.1 and 0.0.2.
> Full support expected in 0.0.3.

```python
from ragu.graph.types import Entity

entity = Entity(
    entity_name="Python",
    entity_type="Language",
    description="A programming language.",
    source_chunk_id=["chunk-1"],
)

print(entity.id, entity.to_text())
```

```python
from dataclasses import dataclass

from ragu.graph.types import Entity


@dataclass(slots=True)
class ProductEntity(Entity):
    sku: str = ""
    confidence: float = 1.0


entity = ProductEntity(
    entity_name="RAGU Pro",
    entity_type="Product",
    description="A domain-specific product entity.",
    source_chunk_id=["chunk-1"],
    sku="RAGU-PRO",
    confidence=0.97,
)

print(entity.sku, entity.to_text())
```

### Relation

Directed graph edge dataclass.

- Purpose: semantic edge between two entities.
- Important fields: `subject_id`, `object_id`, `subject_name`, `object_name`, `relation_type`, `description`, `relation_strength`, `source_chunk_id`.
- ID behavior: defaults to `rel-...` hash of subject, object, and relation type.
- Extension point: `Relation` is the base graph model for RAGU relations and can be inherited to add domain-specific edge metadata. Custom relation classes should preserve the `Edge` contract used by storage adapters.

> NOTE: custom relation extraction is not supported in RAGU versions 0.0.1 and 0.0.2.
> Full support expected in 0.0.3.
>
```python
from ragu.graph.types import Entity, Relation

python = Entity(
    entity_name="Python",
    entity_type="Language",
    description="A programming language.",
    source_chunk_id=["chunk-1"]
)
guido = Entity(
    entity_name="Guido van Rossum",
    entity_type="Person",
    description="Creator of Python.",
    source_chunk_id=["chunk-1"]
)
relation = Relation(
    subject_id=guido.id,
    object_id=python.id,
    subject_name=guido.entity_name,
    object_name=python.entity_name,
    relation_type="CREATED",
    description="Guido van Rossum created Python.",
)

print(relation.id, relation.to_text())
```

```python
from dataclasses import dataclass

from ragu.graph.types import Entity, Relation


@dataclass(slots=True)
class EvidenceRelation(Relation):
    evidence_quote: str = ""
    extractor_name: str = ""


python = Entity(
    entity_name="Python",
    entity_type="Language",
    description="A programming language.",
    source_chunk_id=["chunk-1"]
)

# "Short" creation
guido = Entity(
    "Guido van Rossum",
    "Person",
    "Creator of Python.",
    ["chunk-1"]
)
relation = EvidenceRelation(
    subject_id=guido.id,
    object_id=python.id,
    subject_name=guido.entity_name,
    object_name=python.entity_name,
    relation_type="CREATED",
    description="Guido van Rossum created Python.",
    evidence_quote="Python was created by Guido van Rossum.",
    extractor_name="two_stage_llm",
)

print(relation.evidence_quote)
```

### BuilderArguments

Configuration for build behavior.

- `use_llm_summarization`: summarize duplicate descriptions with LLM.
- `use_clustering`: cluster similar entities before summarization.
- `build_only_vector_context`: skip graph artifact extraction and store chunks only.
- `make_community_summary`: run community detection and summarization.
- `remove_isolated_nodes`: add `RemoveIsolatedNodes` post-processor.
- `vectorize_chunks`: **currently a no-op**, kept for backward compatibility. Chunk vectorization always happens inside `Index.upsert_chunks` regardless of this value.

#### Pipeline preset: chunk-vector index only

```python
import asyncio

from ragu import KnowledgeGraph, SimpleChunker
from ragu.graph.graph_builder_pipeline import BuilderArguments
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
        chunker=SimpleChunker(max_chunk_size=800, overlap=80),
        builder_settings=BuilderArguments(
            build_only_vector_context=True,
            make_community_summary=False,
        ),
    )

    await graph.build_from_docs(["RAGU can index chunks without extracting a graph."])
    print(await graph.index.chunks_kv_storage.all_keys())


asyncio.run(main())
```

This preset is for naive vector RAG. It stores chunks and chunk vectors, but skips entity/relation extraction, graph edges, communities, and community summaries.

#### Pipeline preset: fast graph extraction without LLM summarization

```python
import asyncio

from ragu import KnowledgeGraph, SimpleChunker
from ragu.graph.graph_builder_pipeline import BuilderArguments
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI
from ragu.triplet import TwoStageArtifactsExtractorLLM


async def main():
    client = CachedAsyncOpenAI(
        base_url="https://api.openai.com/v1",
        api_key="dummy-api-token",
    )
    llm = LLMOpenAI(client=client, model_name="gpt-4o-mini")
    embedder = EmbedderOpenAI(
        client=client,
        model_name="text-embedding-3-small",
        dim=1536,
    )
    graph = KnowledgeGraph(
        llm=llm,
        embedder=embedder,
        chunker=SimpleChunker(max_chunk_size=1200, overlap=100),
        artifact_extractor=TwoStageArtifactsExtractorLLM(llm),
        builder_settings=BuilderArguments(
            use_llm_summarization=False,
            use_clustering=False,
            make_community_summary=False,
            remove_isolated_nodes=True,
        ),
    )

    await graph.build_from_docs(["Guido van Rossum created Python."])
    print(await graph.index.graph_backend.get_all_nodes())


asyncio.run(main())
```

This preset still extracts entities and relations with an LLM, but duplicate descriptions are merged without additional LLM summarization and no community summaries are generated.

#### Pipeline preset: full GraphRAG with community summaries

```python
import asyncio

from ragu import KnowledgeGraph, SimpleChunker
from ragu.graph.graph_builder_pipeline import BuilderArguments
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI
from ragu.triplet import TwoStageArtifactsExtractorLLM


async def main():
    client = CachedAsyncOpenAI(
        base_url="https://api.openai.com/v1",
        api_key="dummy-api-token",
    )
    llm = LLMOpenAI(client=client, model_name="gpt-4o-mini")
    embedder = EmbedderOpenAI(
        client=client,
        model_name="text-embedding-3-small",
        dim=1536,
    )
    graph = KnowledgeGraph(
        llm=llm,
        embedder=embedder,
        chunker=SimpleChunker(max_chunk_size=1200, overlap=100),
        artifact_extractor=TwoStageArtifactsExtractorLLM(
            llm,
            do_entity_validation=True,
            do_relation_validation=True,
        ),
        builder_settings=BuilderArguments(
            use_llm_summarization=True,
            use_clustering=False,
            make_community_summary=True,
            remove_isolated_nodes=True,
            summarize_only_if_more_than=7,
            max_cluster_size=128,
            random_seed=42,
        ),
    )

    await graph.build_from_docs(["Python was created by Guido van Rossum."])
    print(await graph.index.community_summary_kv_storage.all_keys())


asyncio.run(main())
```

This preset is the default GraphRAG path: extract artifacts, merge/summarize duplicate descriptions, remove isolated entities, detect Leiden communities, and write community summaries for global search.

#### Pipeline preset: large-corpus entity clustering before summarization

```python
import asyncio

from ragu import KnowledgeGraph, SimpleChunker
from ragu.graph.graph_builder_pipeline import BuilderArguments
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI
from ragu.triplet import ArtifactsExtractorLLM


async def main():
    client = CachedAsyncOpenAI(
        base_url="https://api.openai.com/v1",
        api_key="dummy-api-token",
    )
    llm = LLMOpenAI(client=client, model_name="gpt-4o-mini")
    embedder = EmbedderOpenAI(
        client=client,
        model_name="text-embedding-3-small",
        dim=1536,
    )
    graph = KnowledgeGraph(
        llm=llm,
        embedder=embedder,
        chunker=SimpleChunker(max_chunk_size=1500, overlap=150),
        artifact_extractor=ArtifactsExtractorLLM(llm, do_validation=True),
        builder_settings=BuilderArguments(
            use_llm_summarization=True,
            use_clustering=True,
            cluster_only_if_more_than=500,
            summarize_only_if_more_than=5,
            make_community_summary=True,
            max_cluster_size=256,
        ),
    )

    await graph.build_from_docs(["A long corpus split into many chunks."])
    print(await graph.index.graph_backend.get_all_nodes())


asyncio.run(main())
```

This preset is intended for many repeated entity mentions. Entity descriptions are embedded and clustered before LLM summarization when the duplicate-description count crosses `cluster_only_if_more_than`.

### InMemoryGraphBuilder

In-memory graph build pipeline.

- Purpose: calls artifact extractor, entity/relation summarizers, extra modules, community clustering, and community summarizer.
- Important methods: `extract_graph`, `cluster_graph`.

```python
import asyncio

from ragu.graph.graph_builder_pipeline import BuilderArguments, InMemoryGraphBuilder
from ragu.graph.types import Entity, Relation
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.openai import CachedAsyncOpenAI


async def main():
    client = CachedAsyncOpenAI(base_url="https://api.openai.com/v1", api_key="dummy-api-token")
    embedder = EmbedderOpenAI(client=client, model_name="text-embedding-3-small", dim=1536)
    builder = InMemoryGraphBuilder(
        embedder=embedder,
        build_parameters=BuilderArguments(build_only_vector_context=True),
    )
    python = Entity("Python", "Language", "A programming language.", ["chunk-1"])
    guido = Entity("Guido van Rossum", "Person", "Creator of Python.", ["chunk-1"])
    relation = Relation(guido.id, python.id, guido.entity_name, python.entity_name, "CREATED", "Created Python.")
    communities = await builder.cluster_graph([python, guido], [relation])
    print(communities)


asyncio.run(main())
```

### Index

Storage coordinator.

- Purpose: keeps graph backend, vector DBs, chunk KV, community KV, and summary KV in sync.
- Important methods: `upsert_nodes`, `upsert_edges`, `upsert_chunks`, `delete_chunks`, `a_read_lock`, `check_consistency`.
- Coordinated reads and updates also include `get_summaries`, `delete_summaries`, `update_communities`, `update_summaries`, and `edges_degrees`; the matching `KnowledgeGraph` methods delegate to them.
- Storage defaults: `NetworkXStorage`, `JsonKVStorage`, `NanoVectorDBStorage`.

```python
import asyncio

from ragu.graph.index import Index, StorageArguments
from ragu.graph.types import Entity, Relation
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.openai import CachedAsyncOpenAI


async def main():
    client = CachedAsyncOpenAI(base_url="https://api.openai.com/v1", api_key="dummy-api-token")
    embedder = EmbedderOpenAI(client=client, model_name="text-embedding-3-small", dim=1536)
    index = Index(arguments=StorageArguments(), embedder=embedder)
    entity = Entity("Python", "Language", "A programming language.", ["chunk-1"])
    await index.upsert_nodes([entity])
    print(await index.get_nodes([entity.id]))


asyncio.run(main())
```

### GraphRetriever

Query-time vector helper.

- Purpose: build dense/sparse query vectors and resolve vector hits to entities, relations, or chunks.
- Important methods: `query_entities`, `query_relations`, `query_chunks`.

```python
import asyncio

from ragu.graph.graph_retrieve_backend import GraphRetriever
from ragu.graph.knowledge_graph import KnowledgeGraph
from ragu.graph.graph_builder_pipeline import BuilderArguments
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.openai import CachedAsyncOpenAI


async def main():
    client = CachedAsyncOpenAI(base_url="https://api.openai.com/v1", api_key="dummy-api-token")
    embedder = EmbedderOpenAI(client=client, model_name="text-embedding-3-small", dim=1536)
    graph = KnowledgeGraph(
        llm=None,
        embedder=embedder,
        builder_settings=BuilderArguments(build_only_vector_context=True),
    )
    retriever = GraphRetriever(knowledge_graph=graph, embedder=embedder)
    point = await retriever.build_query_vectors("Python")
    print(point.dense_embedding is not None)


asyncio.run(main())
```

## Concurrent Access

`KnowledgeGraph` and its `Index` share element-level reader/writer locks. Reads of the same elements can run together. A write waits for readers and writers of its affected elements, while operations on unrelated elements can continue. Synchronization is implemented above the storage adapters and is enabled automatically for the facade's CRUD methods.

A write scope covers validation, reading existing values, merging, vectorization, graph/vector/KV changes, reverse-index updates, and persistence callbacks. Deleting a node also locks its incident relations and their endpoints until the corresponding graph and vector records have been removed. Deleting a chunk locks the referring nodes and relations, including incident relations of the deleted nodes. Edge writes protect their endpoints, so a new relation cannot appear during deletion of an endpoint. Community deletion protects its summary as well.

Storage lifecycles start after validation and vectorization. Every successfully started backend is finalized before the element locks are released, including on failure or cancellation. A batch's start, writes, and finalization run in the same asyncio task, preserving task-local adapter state. Batches on unrelated elements may overlap, so adapters must support concurrent batches when shared this way.

The affected set is discovered before acquiring the locks. RAGU acquires that set atomically, then checks it again; if the cascade has grown, it releases the set and retries. A waiting writer delays later conflicting readers, while unrelated requests can pass it. This avoids acquiring only part of a cascade and waiting for another operation that holds the remaining elements.

### Keeping Multiple Reads Consistent

Each read method protects its own operation and returns a detached snapshot. Use `a_read_lock` to keep selected elements stable across several awaited reads:

```python
from ragu import KnowledgeGraph
from ragu.graph.types import Entity


async def read_pair(
    graph: KnowledgeGraph,
    first_id: str,
    second_id: str,
) -> tuple[Entity | None, Entity | None]:
    async with graph.a_read_lock(node_ids=[first_id, second_id]):
        first = (await graph.get_entities([first_id]))[0]
        second = (await graph.get_entities([second_id]))[0]
        return first, second
```

`KnowledgeGraph.a_read_lock` delegates to `Index.a_read_lock`; both accept the following keyword arguments:

| Argument | Protected data |
| --- | --- |
| `node_ids=()` | Selected nodes. |
| `edge_specs=()` | Relations specified by `(subject_id, object_id, relation_id)` and their endpoints; `relation_id=None` selects the whole directed pair. |
| `chunk_ids=()` | Selected source chunks. |
| `community_ids=()` | Selected communities and the elements needed to reconstruct them. |
| `summary_ids=()` | Selected community summaries. |
| `include_neighbors=False` | When enabled, expands selected nodes to their retrieval neighborhood, including relations, neighboring nodes, source chunks, communities, and summaries. |
| `all_summaries=False` | When enabled, protects all community summaries and the community metadata used to filter them. |

The context manager is asynchronous: use `async with`, and perform the protected calls in the same task. Nested scopes can reuse an already protected subset. Expanding a held set or upgrading a read lock to a write lock raises `ValueError`; include all required IDs in the outer scope instead. Locks are released on normal completion, exceptions, and cancellation.

`GraphRetriever` embeds each query batch once, discovers vector candidates, and repeats the vector search under their element read locks. If the repeated search returns additional candidates, it releases the scope and retries with the expanded set. This keeps returned scores and vector metadata consistent with the resolved graph or chunk records. A stable selection requires two vector queries.

`LocalSearchEngine` keeps its selected neighborhood protected through reranking and context assembly. If an anchor changes between vector retrieval and acquiring the neighborhood scope, it repeats retrieval before assembling the context. It releases the read scope before generating the LLM answer. Global search protects summary collection while loading its context; naive search resolves matched chunk IDs under their read scope. Other elements remain available for online updates during these reads. Whole-index operations, such as consistency checking, reindexing, and closing, use a wider scope because they inspect or replace data across the index.

### Scope of the Guarantee

Share one `KnowledgeGraph`/`Index` instance within one process. The lock manager supports concurrent asyncio tasks and threads with separate event loops; a chosen adapter or model client may impose additional restrictions on sharing its own connections across threads or loops. Independently constructed indexes and separate processes do not share locks.

RAGU also guards individual adapter calls and persistence callbacks so that shared-memory backends can safely inspect or modify their internal structures. These short guards do not cover embedding or LLM calls. Direct access to `index.graph_backend`, vector stores, or KV stores bypasses the element-level protocol; use the facade methods or an appropriate read scope for compound reads, and use facade methods for mutations.

Locks coordinate concurrent access; they do not provide transactional rollback across graph, vector, and KV stores. A backend failure or cancellation can still require the existing consistency checks and recovery procedures.

The conceptual workflow is described in [English](../../docs/en/ragu_components.md#concurrent-reads-and-online-updates) and [Russian](../../docs/ru/ragu_components.md#конкурентное-чтение-и-онлайн-обновления).

## Data Flow

Input: `list[str]` documents or explicit `Entity` and `Relation` objects.

Output:

- graph backend nodes and directed multigraph edges
- vector records for entities, relations, and chunks
- KV records for chunks, communities, and community summaries

Used by:

- `ragu.search_engine.LocalSearchEngine`
- `ragu.search_engine.GlobalSearchEngine`
- `ragu.search_engine.NaiveSearchEngine`
- `ragu.storage` adapters

## Usage Examples

### Example 1 - Minimal usage

```python
import asyncio

from ragu.graph.knowledge_graph import KnowledgeGraph
from ragu.graph.graph_builder_pipeline import BuilderArguments
from ragu.graph.types import Entity, Relation
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
        builder_settings=BuilderArguments(build_only_vector_context=True),
    )

    python = Entity("Python", "Language", "A programming language.", ["chunk-1"])
    guido = Entity("Guido van Rossum", "Person", "Creator of Python.", ["chunk-1"])
    relation = Relation(
        subject_id=guido.id,
        object_id=python.id,
        subject_name=guido.entity_name,
        object_name=python.entity_name,
        relation_type="CREATED",
        description="Guido van Rossum created Python.",
        source_chunk_id=["chunk-1"],
    )

    await graph.upsert_entities([python, guido])
    await graph.upsert_relations([relation])

    stored = await graph.get_entities([python.id, guido.id])
    print([entity.entity_name for entity in stored if entity])


asyncio.run(main())
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
        chunker=SimpleChunker(max_chunk_size=100),
        builder_settings=BuilderArguments(build_only_vector_context=True),
    )

    await graph.build_from_docs(["RAGU supports naive vector search over chunks."])
    chunk_ids = await graph.index.chunks_kv_storage.all_keys()
    print(chunk_ids)


asyncio.run(main())
```

## Integration Points

- LLMs: used by artifact extraction, entity/relation summarization, and community summarization.
- Embedders: `Index` embeds entity text, relation text, and chunk content before vector upsert.
- Sparse embedders: when provided, `Index` stores sparse vectors and `GraphRetriever` builds sparse query vectors.
- Qdrant: configure `StorageArguments(vdb_storage_type=QdrantVectorDBStorage, vdb_storage_kwargs={...})`.
- Search engines: consume `KnowledgeGraph` plus embedders through `GraphRetriever`.

## Configuration

Key build settings:

- `BuilderArguments(build_only_vector_context=True)`: chunk-only vector index, useful for `NaiveSearchEngine`.
- `BuilderArguments(make_community_summary=True)`: enables Leiden clustering plus LLM summaries.
- `BuilderArguments(remove_isolated_nodes=True)`: filters nodes without relations through `RemoveIsolatedNodes`.

Key storage settings:

```python
from ragu.graph.index import StorageArguments
from ragu.storage.vdb_storage_adapters.qdrant_vdb import QdrantVectorDBStorage

storage = StorageArguments(
    vdb_storage_type=QdrantVectorDBStorage,
    vdb_storage_kwargs={"location": ":memory:", "sparse_type": "bm25"},
)
```

## Dependencies

Internal:

- `ragu.chunker`
- `ragu.triplet`
- `ragu.models`
- `ragu.storage`
- `ragu.utils.token_truncation`

External:

- `networkx`
- `graspologic_native`
- `pandas`
- `numpy`

## Notes / Pitfalls

- Graph storage assumes directed multigraph semantics; edge identity is `(subject_id, object_id, relation_id)`.
- Edges cannot be inserted before their endpoint entities exist.
- `upsert_entities` and `upsert_relations` reject duplicate IDs in the same request, then merge with existing stored items.
- Community detection builds an undirected temporary graph for Leiden clustering, then stores communities separately in KV storage.
- `build_only_vector_context=True` requires an embedder but does not require an LLM or artifact extractor.
