# Module: ragu.storage

## Role in RAGU Pipeline

`ragu.storage` persists the outputs of graph building and supplies vector search for retrieval. It contains abstract contracts plus concrete graph, KV, and vector storage adapters.

Pipeline position:

```text
Entity/Relation/Chunk/Community -> Index -> graph storage + KV storage + vector storage
query vector -> vector storage -> EmbeddingHit -> graph/KV resolution
```

## Overview

The module exists to keep `KnowledgeGraph` independent from a specific backend. RAGU can use local NetworkX/GML and JSON/Nano vector files for development, or Qdrant for dense and hybrid retrieval.

## Key Components

### Node

Base graph node protocol from `ragu.storage.types`.

- Purpose: defines the minimum shape graph storage adapters must support for node-like objects.
- Required fields by convention:
  - `id: str`
  - `source_chunk_id: list[str]`
  - `clusters: list[ClusterInfo]`
- Important methods:
  - `to_dict()`: serialize the node payload.
  - `to_text()`: produce text used by vectorization or fallback display.
- Common child class: `ragu.graph.types.Entity`.
- Model extension reminder: `Entity` itself is designed to be inherited for richer domain entities. Storage adapters should operate on the configured child class, not on `Entity` specifically.
- Adapter requirement: graph storages must not hard-code `Entity`. They receive `node_cls` from `Index` and must be able to store and reconstruct any dataclass-like child class of `Node` that follows the required fields.

```python
from ragu.graph.types import Entity
from ragu.storage.types import Node

entity = Entity("Python", "Language", "A programming language.", ["chunk-1"])

print(isinstance(entity, Node))
print(entity.to_dict())
print(entity.to_text())
```

### Edge

Base graph edge protocol from `ragu.storage.types`.

- Purpose: defines the minimum shape graph storage adapters must support for edge-like objects.
- Required fields by convention:
  - `id: str`
  - `subject_id: str`
  - `object_id: str`
  - `source_chunk_id: list[str]`
- Important methods:
  - `to_dict()`: serialize the edge payload.
  - `to_text()`: produce text used by vectorization or fallback display.
- Common child class: `ragu.graph.types.Relation`.
- Model extension reminder: `Relation` itself is designed to be inherited for richer domain relations. Storage adapters should operate on the configured child class, not on `Relation` specifically.
- Adapter requirement: graph storages must not hard-code `Relation`. They receive `edge_cls` from `Index` and must be able to store and reconstruct any dataclass-like child class of `Edge` that follows the required fields.

```python
from ragu.graph.types import Entity, Relation
from ragu.storage.types import Edge

python = Entity("Python", "Language", "A programming language.", ["chunk-1"])
guido = Entity("Guido van Rossum", "Person", "Creator of Python.", ["chunk-1"])
relation = Relation(
    subject_id=guido.id,
    object_id=python.id,
    subject_name=guido.entity_name,
    object_name=python.entity_name,
    relation_type="CREATED",
    description="Guido van Rossum created Python.",
)

print(isinstance(relation, Edge))
print(relation.to_dict())
print(relation.to_text())
```

### BaseStorage

Common lifecycle contract for every storage backend.

- Purpose: gives `Index` a uniform way to notify storage backends before indexing, after indexing, and after query work.
- Used by: graph, KV, and vector storage base classes.
- Important hooks:
  - `index_start_callback()`: optional setup before an indexing batch.
  - `index_done_callback()`: flush, persist, or finalize newly indexed data.
  - `query_done_callback()`: optional cleanup after query-time reads.
- Implementation expectation: adapters may no-op these hooks, but they must expose them so orchestration code can treat all storage backends consistently.

```python
from ragu.storage.base_storage import BaseStorage

print(BaseStorage.__abstractmethods__)
```

### BaseGraphStorage

Directed multigraph storage contract.

- Purpose: store nodes and edges with edge specs `(subject_id, object_id, relation_id)`.
- Generic parameters:
  - `NodeT`: a subclass of `ragu.storage.types.Node`.
  - `EdgeT`: a subclass of `ragu.storage.types.Edge`.
- Important read methods:
  - `get_nodes(node_ids)`: ordered node lookup with `None` for misses.
  - `get_edges(edge_specs)`: ordered edge lookup by `(subject_id, object_id, relation_id)`.
  - `get_all_nodes()`, `get_all_edges()`: full graph scans.
  - `get_all_edges_for_nodes(node_ids)`: incident-edge lookup for local search and cascade deletion.
  - `edges_degrees(edge_specs)`: degree signal for relation ranking.
- Important write methods:
  - `upsert_nodes(nodes)`, `delete_nodes(node_ids)`.
  - `upsert_edges(edges)`, `delete_edges(edge_specs)`.
- Invariant: RAGU treats graph storage as a directed multigraph; the relation ID is part of edge identity.
- Implementation expectation: an adapter must preserve dataclass payloads well enough to reconstruct the configured `node_cls` and `edge_cls`.
- Subclass requirement: storage adapters must operate with any child class of `Node` and `Edge`, not only the built-in `Entity` and `Relation` classes.

```python
from ragu.storage.base_storage import BaseGraphStorage, EdgeSpec
from ragu.storage.types import Edge, Node

edge_spec: EdgeSpec = ("subject-id", "object-id", "relation-id")
print(BaseGraphStorage[Node, Edge].__abstractmethods__)
print(edge_spec)
```

### BaseKVStorage

Key-value storage contract.

- Purpose: store chunks, communities, and summaries.
- Generic parameter: `T`, the value type stored under string IDs.
- Important read methods:
  - `all_keys()`: list all stored keys.
  - `get_by_id(id)`: fetch one value.
  - `get_by_ids(ids, fields=None)`: ordered batch lookup with optional field projection.
  - `filter_keys(data)`: return keys from the input list that are missing in storage.
- Important write methods:
  - `upsert(data)`: insert or replace values by key.
  - `delete(ids)`: delete keys.
  - `drop()`: clear backend contents.
- Implementation expectation: missing keys should resolve to `None`, and batch lookups should preserve input order.

```python
from typing import Any

from ragu.storage.base_storage import BaseKVStorage

print(BaseKVStorage[dict[str, Any]].__abstractmethods__)
```

### BaseVectorStorage

Vector storage contract.

- Purpose: store dense and optional sparse vectors.
- Input value type: `ragu.storage.types.Point`.
- Sparse vector type: `ragu.storage.types.SparseEmbedding`.
- Query output type: `list[ragu.storage.types.EmbeddingHit]`.
- Important read methods:
  - `query(point, **kwargs)`: nearest-neighbor or hybrid lookup for one query point.
  - `get_all_ids()`: list vector record IDs.
  - `get_points_by_ids(ids)`: ordered point lookup with `None` for misses.
  - `get_payloads_by_ids(ids)`: ordered metadata lookup with `None` for misses.
- Important write methods:
  - `upsert(data)`: insert or replace vector records.
  - `delete(ids)`: remove vector records.
- Implementation expectation: adapters should accept dense-only, sparse-only, or hybrid `Point` objects if the backend mode supports them; unsupported modes should fail clearly or document degraded behavior.

```python
from ragu.storage.base_storage import BaseVectorStorage
from ragu.storage.types import EmbeddingHit, Point

print(BaseVectorStorage.__abstractmethods__)
print(Point)
print(EmbeddingHit)
```

### SparseEmbedding

Sparse vector payload from `ragu.storage.types`.

- Purpose: represent lexical/sparse embeddings from BM25, BM42, SPLADE, or custom sparse encoders.
- Fields:
  - `indices: list[int]`
  - `values: list[float]`
- Invariant: `indices` and `values` must have the same length.
- Used by: `Point.sparse_embedding`, Qdrant sparse vectors, hybrid retrieval.

```python
from ragu.storage.types import SparseEmbedding

sparse = SparseEmbedding(indices=[10, 42], values=[0.7, 1.2])
print(sparse.indices)
print(sparse.values)
```

### Point

Vector database record from `ragu.storage.types`.

- Purpose: carry a vector record into or out of vector storage.
- Fields:
  - `id: str`
  - `dense_embedding: DenseEmbedding | None`
  - `sparse_embedding: SparseEmbedding | None`
  - `metadata: dict[str, Any]`
- Invariant: at least one of `dense_embedding` or `sparse_embedding` must be present.
- Used by: `BaseVectorStorage.upsert`, `BaseVectorStorage.query`, and vector adapter lookup methods.

```python
import numpy as np

from ragu.storage.types import Point, SparseEmbedding

point = Point(
    id="chunk-1",
    dense_embedding=np.array([0.1, 0.2, 0.3]),
    sparse_embedding=SparseEmbedding(indices=[3], values=[0.9]),
    metadata={"content": "RAGU stores vectors."},
)

print(point.id, point.metadata)
```

### EmbeddingHit

Vector search result from `ragu.storage.types`.

- Purpose: return ranked matches from vector storage.
- Fields:
  - `id: str`: matched record ID.
  - `distance: float`: backend score or distance.
  - `metadata: dict[str, Any]`: payload returned by the vector backend.
- Used by: `GraphRetriever` to resolve vector hits back to nodes, edges, or chunks.

```python
from ragu.storage.types import EmbeddingHit

hit = EmbeddingHit(
    id="chunk-1",
    distance=0.93,
    metadata={"doc_id": "doc-1"},
)

print(hit.id, hit.distance, hit.metadata)
```

### NetworkXStorage

Local graph backend.

- Purpose: store graph data in a NetworkX directed multigraph and persist to GML.
- Used by default through `StorageArguments`.

```python
import tempfile

from ragu.graph.types import Entity, Relation
from ragu.storage.graph_storage_adapters.networkx_adapter import NetworkXStorage

with tempfile.TemporaryDirectory() as directory:
    storage = NetworkXStorage(f"{directory}/kg.gml", node_cls=Entity, edge_cls=Relation)
    print(storage)
```

### JsonKVStorage

JSON-backed KV adapter.

- Purpose: local persistent dictionaries for chunks, communities, and summaries.

```python
import asyncio
import tempfile

from ragu.storage.kv_storage_adapters.json_storage import JsonKVStorage


async def main():
    with tempfile.TemporaryDirectory() as directory:
        storage = JsonKVStorage(storage_folder=directory, filename="data.json")
        await storage.upsert({"id": {"value": 1}})
        await storage.index_done_callback()
        print(await storage.all_keys())


asyncio.run(main())
```

### NanoVectorDBStorage

Default local vector DB adapter.

- Purpose: lightweight dense vector storage for local development and tests.

```python
import tempfile

from ragu.storage.vdb_storage_adapters.nano_vdb import NanoVectorDBStorage

with tempfile.TemporaryDirectory() as directory:
    storage = NanoVectorDBStorage(
        embedding_dim=3,
        storage_folder=directory,
        filename="vectors.json",
    )
    print(storage.embedding_dim)
```

### QdrantVectorDBStorage

Qdrant-backed vector adapter.

- Purpose: dense-only or dense+sparse hybrid retrieval.
- Important parameters: `embedding_dim`, `collection_name`, `location`, `url`, `sparse_type`.
- Sparse modes: `bm25`, `bm42`, `splade`, `custom`.

```python
from ragu.storage.vdb_storage_adapters.qdrant_vdb import QdrantVectorDBStorage

storage = QdrantVectorDBStorage(
    embedding_dim=1536,
    location=":memory:",
    collection_name="ragu_demo",
)

print(storage.collection_name)
```

### Vector Payload Summary

Compact summary of the vector payload dataclasses.

- `Point`: ID plus dense and/or sparse vector and metadata.
- `SparseEmbedding`: aligned `indices` and `values`.
- `EmbeddingHit`: vector query hit with `id`, `distance`, and metadata.

```python
import numpy as np

from ragu.storage.types import EmbeddingHit, Point, SparseEmbedding

sparse = SparseEmbedding(indices=[1, 42], values=[0.5, 1.0])
point = Point(id="p1", dense_embedding=np.array([1.0, 0.0, 0.0]), sparse_embedding=sparse)
hit = EmbeddingHit(id=point.id, distance=0.99, metadata={"kind": "entity"})

print(hit)
```

## Data Flow

Input:

- `Entity` and `Relation` objects from graph building
- `Chunk` objects from chunking
- `Community` and `CommunitySummary` from clustering
- `Point` objects from `Index`

Output:

- stored graph nodes and multigraph edges
- stored KV records
- vector query hits used by `GraphRetriever`

Used by:

- `ragu.graph.Index`
- `ragu.graph.KnowledgeGraph`
- `ragu.search_engine`

## Usage Examples

### Example 1 - Minimal usage

```python
import asyncio
import tempfile

from ragu.graph.types import Entity, Relation
from ragu.storage.graph_storage_adapters.networkx_adapter import NetworkXStorage


async def main():
    with tempfile.TemporaryDirectory() as directory:
        storage = NetworkXStorage(
            filename=f"{directory}/knowledge_graph.gml",
            node_cls=Entity,
            edge_cls=Relation,
        )

        python = Entity("Python", "Language", "A programming language.", ["chunk-1"])
        guido = Entity("Guido van Rossum", "Person", "Creator of Python.", ["chunk-1"])
        created = Relation(
            subject_id=guido.id,
            object_id=python.id,
            subject_name=guido.entity_name,
            object_name=python.entity_name,
            relation_type="CREATED",
            description="Guido van Rossum created Python.",
            source_chunk_id=["chunk-1"],
        )

        await storage.upsert_nodes([python, guido])
        await storage.upsert_edges([created])

        nodes = await storage.get_nodes([python.id, guido.id])
        edges = await storage.get_edges([(guido.id, python.id, created.id)])
        degrees = await storage.edges_degrees([(guido.id, python.id, created.id)])

        print(nodes)
        print(edges)
        print(degrees)


asyncio.run(main())
```

### Example 2 - Vector DB usage

```python
import asyncio
import numpy as np

from ragu.storage.types import Point
from ragu.storage.vdb_storage_adapters.qdrant_vdb import QdrantVectorDBStorage


async def main():
    store = QdrantVectorDBStorage(
        embedding_dim=3,
        location=":memory:",
        collection_name="readme_vectors",
    )

    await store.upsert([
        Point(id="doc-1", dense_embedding=np.array([1.0, 0.0, 0.0]), metadata={"text": "RAGU"}),
    ])

    hits = await store.query(
        Point(id="query", dense_embedding=np.array([1.0, 0.0, 0.0])),
        top_k=1,
    )
    print(hits[0].id)


asyncio.run(main())
```

### Example 3 - Vector DB payload lookup and delete

```python
import asyncio
import numpy as np

from ragu.storage.types import Point
from ragu.storage.vdb_storage_adapters.qdrant_vdb import QdrantVectorDBStorage


async def main():
    storage = QdrantVectorDBStorage(
        embedding_dim=3,
        location=":memory:",
        collection_name="readme_payload_vectors",
    )
    await storage.upsert([
        Point(
            id="chunk-1",
            dense_embedding=np.array([0.1, 0.2, 0.3]),
            metadata={"content": "RAGU stores chunk vectors.", "doc_id": "doc-1"},
        )
    ])

    payloads = await storage.get_payloads_by_ids(["chunk-1"])
    points = await storage.get_points_by_ids(["chunk-1"])
    print(payloads)
    print(points)

    await storage.delete(["chunk-1"])
    print(await storage.get_payloads_by_ids(["chunk-1"]))


asyncio.run(main())
```

### Example 4 - Pipeline usage

```python
import asyncio

from ragu import BuilderArguments, KnowledgeGraph, StorageArguments
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.openai import CachedAsyncOpenAI
from ragu.storage.vdb_storage_adapters.qdrant_vdb import QdrantVectorDBStorage


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
        storage_settings=StorageArguments(
            vdb_storage_type=QdrantVectorDBStorage,
            vdb_storage_kwargs={"location": ":memory:"},
        ),
    )
    await graph.build_from_docs(["RAGU can use Qdrant for chunk vectors."])
    print(await graph.index.chunks_kv_storage.all_keys())


asyncio.run(main())
```

## Integration Points

- Embedders: `Index` creates dense embeddings before vector upsert.
- Sparse embedders: `Index` stores sparse document vectors; `GraphRetriever` sends sparse query vectors.
- Qdrant: hybrid mode stores one dense vector field named `"dense"` and one sparse vector field named by `sparse_type`.
- Graph module: `StorageArguments` selects storage implementations.

## Configuration

Default storage is controlled by `ragu.graph.index.StorageArguments`:

- `graph_backend_storage=NetworkXStorage`
- `kv_storage_type=JsonKVStorage`
- `vdb_storage_type=NanoVectorDBStorage`

Qdrant deployment modes:

- local on-disk: omit remote arguments and pass `storage_folder`/`filename`
- in-memory: `location=":memory:"`
- remote: pass `url`, `host`, `port`, `grpc_port`, and/or `api_key`

Hybrid Qdrant storage additionally needs a matching sparse embedder:

```python
from ragu.models.sparse_embedder import BM25

sparse_embedder = BM25()
vdb_kwargs = {"location": ":memory:", "sparse_type": "bm25"}
```

## Dependencies

Internal:

- `ragu.storage.types`
- `ragu.common.global_parameters.Settings`
- `ragu.utils.ragu_utils`

External:

- `networkx`
- `qdrant-client`
- `numpy`
- `pydantic`

## Notes / Pitfalls

- `Point` must contain at least one dense or sparse embedding.
- `SparseEmbedding.indices` and `SparseEmbedding.values` must have the same length.
- Qdrant validates existing collection schema; mismatched vector dimensions or sparse configuration raise `ValueError`.
- For BM25 and BM42, Qdrant sparse vectors use the `IDF` modifier.
- Deleting chunks through `Index.delete_chunks()` cascades to entities and relations sourced from those chunks.
