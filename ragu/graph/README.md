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
- Important methods: `build_from_docs`, `upsert_entities`, `upsert_relations`, `get_entities`, `get_relations`, `get_chunks`, `reindex_community`, `reindex_descriptions`, `reindex_graph`.
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
- Important methods: `upsert_nodes`, `upsert_edges`, `upsert_chunks`, `delete_chunks`, `check_consistency`.
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

## Table and formula metanodes

`KnowledgeGraph.build_from_docs()` accepts strings or `Document` objects and awaits
`BaseChunker.a_split()`. Existing synchronous chunkers inherit an async adapter;
`RecursiveMarkdownChunker` and `SemanticMarkdownChunker` can also generate object
descriptions. Before calling the chunker, the graph filters processed documents
by a deterministic hash of their content, document ID, title and structured
content. Equivalent strings and default `Document` inputs share a hash; the
local image folder path is excluded. A separate `kv_documents.json` registry
stores completion hashes and their chunk IDs after successful persistence.
Repeated inputs within a batch are also filtered. Chunk-ID deduplication remains
a second step after chunking. Deleting a referenced chunk invalidates the
corresponding document hashes, allowing the document to be ingested again.
The registry is opened lazily so read-only access to a legacy index does not
create new files. Override its backend arguments through
`StorageArguments.documents_kv_storage_kwargs`; it uses the configured
`kv_storage_type`. Existing indexes without document hashes establish the
registry when documents are next ingested.

The graph builder extracts ordinary entities only from text chunks. Each
`TableChunk` / `FormulaChunk` becomes an ordinary `Entity` with type `TABLE` /
`FORMULA`, a chunk-derived `meta-` ID and a `source_chunk_id` pointing to
its original typed chunk. These object entities bypass entity-name merging.
Images and code remain retrievable chunks without automatic graph nodes.

`NearbyRelationExtractor.a_extract_nearby(chunks, entities, relations)` links these
objects to the merged text subgraph. `BuilderArguments.metanode_context_window`
(default `1`, positive integer) selects that many text chunks on each side,
within the same document and section. The model receives surrounding `to_llm()`
text, candidate entity names, IDs and their existing relations. It returns
`source_entity` / `target_entity` names which are resolved against the supplied
nodes. Unknown or ambiguous names, self-links and links between text entities
are rejected. Every accepted link connects the object to one candidate entity.
Sources include the object chunk and all text chunks in the request window;
these identify the supplied context, not model-selected supporting citations.
The stage does not extract entities from table cells or formula contents.

The component is available as `knowledge_graph.pipeline.nearby_relation_extractor`;
customize `table_relations` and `formula_relations` independently using
`update_prompt()`. Each type is processed in a separate batch with its own
system instructions and Jinja2 user template from `default_templates.py`. The template receives
`context.before_chunks` and `context.after_chunks` in source order and renders
them as plain text blocks, without JSON serialization, alongside the
object identity, candidate entities and existing relations. Its schema is
the existing `RelationsExtractionModel`, containing `RelationModel` entries with
`source_entity`, `target_entity`, `relation_type`, `description`, and
`relationship_strength` (0–5). The strength is preserved on the graph relation.
See the [prompt registry](../common/prompts/README.md#default_prompt_templates).
`TABLE` / `FORMULA` and generated object predicates are document structure types,
separate from the text extractor's NEREL vocabulary. No NEREL entries are changed.

```python
from ragu import (
    ArtifactsExtractorLLM, BuilderArguments, Document, KnowledgeGraph,
    RecursiveMarkdownChunker,
)

# llm and embedder are initialized model clients.
graph = KnowledgeGraph(
    llm=llm,
    embedder=embedder,
    chunker=RecursiveMarkdownChunker(llm=llm),
    artifact_extractor=ArtifactsExtractorLLM(llm=llm, embedder=embedder),
    builder_settings=BuilderArguments(metanode_context_window=1),
)
await graph.build_from_docs([Document.from_mineru("data/parsed/report")])
```

The linking stage runs before graph-builder modules and community detection.
The built-in `RemoveIsolatedNodes` preserves object entities even without links.
Without an LLM only the deterministic object nodes are created. Failed linking
calls leave objects intact; oversized requests are skipped using the tokenizer
and context limit from `Settings`, with a warning. Vector-only mode stores typed
chunks and their vectors without graph nodes or nearby-link extraction.

Dense and sparse chunk embeddings use `to_embed()`. Chunk reranking also uses
`to_embed()`, while text extraction and Local/Naive answer contexts use
`to_llm()`. Local search additionally loads source chunks of directly connected
metanodes. Chunk IDs deduplicate results reached through several paths.
Deleting an object's source chunk removes its metanode, incident edges and
corresponding vectors using the existing reverse indexes.

Legacy KV records without `chunk_type` remain readable as `Chunk`; their storage
key is authoritative. New records preserve the concrete class and ID. Existing
indexes need rebuilding to obtain typed metanodes; no in-place backfill is run.
Object IDs depend on document ID, object type and content. Identical objects
in one document share an ID and are subject to chunk-ID deduplication. For edited documents, remove old source chunks
before rebuilding; repeated ingestion of unchanged documents is deduplicated.
