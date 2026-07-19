# Module: ragu.storage.graph_storage_adapters

## Role in RAGU Pipeline

This package provides graph backend implementations for `Index`. Graph storage persists entities and relations after extraction and supports local-search neighborhood traversal.

Pipeline position:

```text
Entity/Relation -> BaseGraphStorage adapter -> graph backend -> LocalSearchEngine
```

## Overview

Graph adapters implement `BaseGraphStorage` for different backends while preserving RAGU's directed multigraph contract.

## Key Components

### NetworkXStorage

- Purpose: default local graph backend.
- Persistence: reads and writes GML files.
- Important parameters: `filename`, `node_cls`, `edge_cls`.

### Neo4jStorage

- Purpose: server-backed graph store for larger graphs and concurrent access.
- The `neo4j` driver is a regular dependency, installed with the package.
- Import it by its full path; it is not re-exported from this package, so that
  the driver stays swappable for a build that strips it out:

  ```python
  from ragu.storage.graph_storage_adapters.neo4j_adapter import Neo4jStorage
  ```

- Important parameters: `uri`, `user`, `password`, `node_cls`, `edge_cls`, `database`.
- Layout: every node carries the shared `:NODE` label plus a per-type label, and
  every edge is written under its own relationship type — both taken from the
  `label_field` the node and edge classes declare (`entity_type` and
  `relation_type` for `Entity`/`Relation`). On the first `index_start_callback()`
  the adapter creates a uniqueness constraint on `:NODE(id)` and an index on the
  node type's `label_field`.
- Relationship types vary per edge, so they cannot serve as a filter. Reads select
  the edges RAGU owns by `r.id IS NOT NULL`; relationships created by other tools
  in the same database are ignored, and unknown properties on an edge are dropped
  rather than raising.

```python
import asyncio

from ragu.graph.index import Index, StorageArguments
from ragu.graph.types import Entity, Relation
from ragu.storage.graph_storage_adapters.neo4j_adapter import Neo4jStorage


async def main():
    storage = Neo4jStorage(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="testpassword",
        node_cls=Entity,
        edge_cls=Relation,
    )
    # Creates the id constraint and the type index; also the first connection attempt.
    await storage.index_start_callback()

    await storage.upsert_nodes([...])
    await storage.upsert_edges([...])

    # Anything the storage interface does not cover: filtering, aggregation,
    # graph algorithms. Pass values as parameters, never format them into the query.
    rows = await storage.run_cypher_query(
        "MATCH (n:NODE) WHERE n.entity_type IN $types RETURN n.id AS id",
        {"types": ["PERSON", "ORGANIZATION"]},
    )

    await storage.close()  # releases the connection pool


asyncio.run(main())
```

To use it as the graph backend of a `KnowledgeGraph`, pass the class and its
arguments through `StorageArguments`, then close the index when done:

```python
index = Index(
    embedder=embedder,
    arguments=StorageArguments(
        graph_backend_storage=Neo4jStorage,
        graph_storage_kwargs={
            "uri": "bolt://localhost:7687",
            "user": "neo4j",
            "password": "testpassword",
        },
    ),
)
...
await index.close()   # closes every backend the index owns
```

## Data Flow

Input: `Entity` nodes and `Relation` edges.

Output: stored nodes, stored multigraph edges, edge-degree values, incident edge lists.

Used by:

- `ragu.graph.Index`
- `ragu.graph.KnowledgeGraph`
- `ragu.search_engine.LocalSearchEngine`

## Usage Examples

### Example 1 - Minimal usage

```python
import asyncio

from ragu.graph.types import Entity, Relation
from ragu.storage.graph_storage_adapters.networkx_adapter import NetworkXStorage


async def main():
    storage = NetworkXStorage(
        filename="knowledge_graph.gml",
        node_cls=Entity,
        edge_cls=Relation,
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

    await storage.upsert_nodes([python, guido])
    await storage.upsert_edges([relation])

    print(await storage.get_nodes([python.id, guido.id]))
    print(await storage.get_edges([(guido.id, python.id, relation.id)]))
    print(await storage.get_all_edges_for_nodes([python.id]))


asyncio.run(main())
```

### Example 2 - Delete graph records

```python
import asyncio

from ragu.graph.types import Entity, Relation
from ragu.storage.graph_storage_adapters.networkx_adapter import NetworkXStorage


async def main():
    storage = NetworkXStorage(
        filename="knowledge_graph.gml",
        node_cls=Entity,
        edge_cls=Relation,
    )
    python = Entity("Python", "Language", "A programming language.", ["chunk-1"])
    guido = Entity("Guido van Rossum", "Person", "Creator of Python.", ["chunk-1"])
    relation = Relation(
        guido.id,
        python.id,
        guido.entity_name,
        python.entity_name,
        "CREATED",
        "Guido van Rossum created Python.",
    )

    await storage.upsert_nodes([python, guido])
    await storage.upsert_edges([relation])
    await storage.delete_edges([(guido.id, python.id, relation.id)])
    await storage.delete_nodes([python.id])

    print(await storage.get_nodes([python.id, guido.id]))


asyncio.run(main())
```

### Example 3 - Pipeline usage

```python
from ragu import BuilderArguments, KnowledgeGraph, StorageArguments
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.openai import CachedAsyncOpenAI
from ragu.storage.graph_storage_adapters.networkx_adapter import NetworkXStorage


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
    storage_settings=StorageArguments(graph_backend_storage=NetworkXStorage),
)
```

## Integration Points

- `Index` calls graph adapters for CRUD and cascade deletion.
- `GraphRetriever` resolves relation vector hits through graph edge specs.
- Local search uses graph edges around retrieved entities.

## Configuration

`StorageArguments.graph_storage_kwargs` is merged with the default `knowledge_graph.gml` filename under `Settings.storage_folder`.

## Dependencies

- `networkx`

## Notes / Pitfalls

- RAGU edge identity is `(subject_id, object_id, relation_id)`.
- `NetworkXStorage.index_done_callback()` writes GML to disk.
- Deleting a node removes connected graph edges.

### Neo4j: filter by property, not by label

Type labels are **additive**. Re-upserting a node under a different type adds the
new label without removing the old one, so a node may carry every type it has
ever had:

```cypher
// after Alice was written as PERSON and later as ORGANIZATION
MATCH (n:NODE {id: 'Alice'}) RETURN labels(n)
// ['NODE', 'PERSON', 'ORGANIZATION']

MATCH (n:PERSON) RETURN n.id, n.entity_type
// [['Alice', 'ORGANIZATION']]   <- stale label still matches
```

The `entity_type` **property** is overwritten on every upsert and is always
current, so filter on it rather than on labels:

```cypher
MATCH (n:NODE {entity_type: 'PERSON'}) RETURN n     // correct, and indexed
MATCH (n:PERSON) RETURN n                           // convenient, may be stale
```

Labels remain useful for eyeballing a graph in Neo4j Browser; they are not a
reliable filter.

This is a deliberate trade. Keying the merge on the label instead
(`MERGE (n:NODE:PERSON {id: ...})`) made a node whose type changed fail to match
its stored counterpart, creating a **second node with the same id** — after which
reads returned an arbitrary one of the two. Accumulated labels are the milder
failure.

In practice RAGU itself never triggers this: `Entity.id` is derived from
`entity_name + entity_type`, so a type change yields a different id, hence a
different node. It is reachable only when ids are assigned explicitly, for
example by user code or by `update_entities`.
