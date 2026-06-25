# Module: ragu.search_engine

## Role in RAGU Pipeline

`ragu.search_engine` is the query layer. It consumes an already built `KnowledgeGraph`, retrieves context with one or more strategies, and optionally asks an LLM to synthesize an answer.

Pipeline position:

```text
KnowledgeGraph + query -> retrieval context -> LLM answer
```

## Overview

The module separates retrieval strategy from indexing. All engines share `BaseEngine`, `a_search()` for retrieval-only calls, and `a_query()` for retrieval plus answer generation.

## Key Components

### BaseEngine

Abstract base for query engines.

- Purpose: stores LLM and context truncation settings.
- Important methods: `a_search`, `a_query`, sync wrappers `search`, `query`.

```python
from ragu.search_engine.base_engine import BaseEngine
from ragu.search_engine.naive_search import NaiveSearchEngine

print(issubclass(NaiveSearchEngine, BaseEngine))  # True — every engine inherits BaseEngine
print([m for m in ("a_search", "a_query", "search", "query") if hasattr(BaseEngine, m)])
```

### SearchEngineRetrieve

Base dataclass for retrieval-only results.

- Purpose: carries `query`, engine-specific `result`, and `metrics`.
- Important method: `to_text()`.

```python
from ragu.search_engine.naive_search import NaiveSearchResult, NaiveSearchRetrieve

retrieval = NaiveSearchRetrieve(
    query="What is RAGU?",
    result=NaiveSearchResult(),
    metrics={"chunks": []},
)

print(retrieval.to_text())
```

### SearchEngineResponse

Generated answer container.

- Purpose: carries `query`, `response`, `retrieval`, and optional `payload`.

```python
from ragu.search_engine.base_engine import SearchEngineResponse
from ragu.search_engine.naive_search import NaiveSearchResult, NaiveSearchRetrieve

retrieval = NaiveSearchRetrieve(query="What is RAGU?", result=NaiveSearchResult())
response = SearchEngineResponse(
    query="What is RAGU?",
    response="RAGU is a GraphRAG engine.",
    retrieval=retrieval,
)

print(str(response))
```

### NaiveSearchEngine

Chunk-vector RAG.

- Purpose: retrieve chunk vectors directly, optionally rerank, then answer.
- Best for: document QA when graph extraction is not needed.
- Uses: `GraphRetriever.query_chunks`.

```python
from ragu import BuilderArguments, KnowledgeGraph, NaiveSearchEngine
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI

client = CachedAsyncOpenAI(base_url="https://api.openai.com/v1", api_key="dummy-api-token")
llm = LLMOpenAI(client=client, model_name="gpt-4o-mini")
embedder = EmbedderOpenAI(client=client, model_name="text-embedding-3-small", dim=1536)
graph = KnowledgeGraph(
    llm=None,
    embedder=embedder,
    builder_settings=BuilderArguments(build_only_vector_context=True),
)
engine = NaiveSearchEngine(llm=llm, knowledge_graph=graph, embedder=embedder)

print(engine.language)
```

### LocalSearchEngine

Graph-neighborhood RAG.

- Purpose: retrieve relevant entities, then collect related relations, chunks, and community summaries.
- Best for: entity-centric questions and local factual neighborhoods.
- Uses: entity vector DB, graph edges, chunk KV, community summary KV.

```python
from ragu import KnowledgeGraph, LocalSearchEngine
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI

client = CachedAsyncOpenAI(base_url="https://api.openai.com/v1", api_key="dummy-api-token")
llm = LLMOpenAI(client=client, model_name="gpt-4o-mini")
embedder = EmbedderOpenAI(client=client, model_name="text-embedding-3-small", dim=1536)
graph = KnowledgeGraph(llm=llm, embedder=embedder)
engine = LocalSearchEngine(llm=llm, knowledge_graph=graph, embedder=embedder)

print(engine.language)
```

### GlobalSearchEngine

Community-summary RAG.

- Purpose: evaluate all community summaries for query relevance and synthesize a global answer.
- Best for: broad questions that require corpus-level themes.
- Requires: community summaries built by `KnowledgeGraph`.

```python
from ragu import GlobalSearchEngine, KnowledgeGraph
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI

client = CachedAsyncOpenAI(base_url="https://api.openai.com/v1", api_key="dummy-api-token")
llm = LLMOpenAI(client=client, model_name="gpt-4o-mini")
embedder = EmbedderOpenAI(client=client, model_name="text-embedding-3-small", dim=1536)
graph = KnowledgeGraph(llm=llm, embedder=embedder)
engine = GlobalSearchEngine(llm=llm, knowledge_graph=graph)

print(engine.language)
```

### MixSearchEngine

Ensemble engine.

- Purpose: run child engines and synthesize combined context or combined answers.
- Important parameter: `ensemble_responses`.

```python
from ragu import BuilderArguments, KnowledgeGraph, MixSearchEngine, NaiveSearchEngine
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI

client = CachedAsyncOpenAI(base_url="https://api.openai.com/v1", api_key="dummy-api-token")
llm = LLMOpenAI(client=client, model_name="gpt-4o-mini")
embedder = EmbedderOpenAI(client=client, model_name="text-embedding-3-small", dim=1536)
graph = KnowledgeGraph(
    llm=None,
    embedder=embedder,
    builder_settings=BuilderArguments(build_only_vector_context=True),
)
naive = NaiveSearchEngine(llm=llm, knowledge_graph=graph, embedder=embedder)
mix = MixSearchEngine(llm=llm, engines=[naive])

print(len(mix.engines))
```

### QueryPlanEngine

Wraps any search engine and decomposes a complex query into dependent subqueries: each subquery is executed via the wrapped engine, intermediate answers are fed into subsequent subqueries, and the partial answers are synthesized into a final response. Useful for multi-hop questions that no single retrieval strategy can answer directly.

```python
from ragu import BuilderArguments, KnowledgeGraph, NaiveSearchEngine, QueryPlanEngine
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI

client = CachedAsyncOpenAI(base_url="https://api.openai.com/v1", api_key="dummy-api-token")
llm = LLMOpenAI(client=client, model_name="gpt-4o-mini")
embedder = EmbedderOpenAI(client=client, model_name="text-embedding-3-small", dim=1536)
graph = KnowledgeGraph(
    llm=None,
    embedder=embedder,
    builder_settings=BuilderArguments(build_only_vector_context=True),
)
engine = NaiveSearchEngine(llm=llm, knowledge_graph=graph, embedder=embedder)
planner = QueryPlanEngine(engine)

print(planner.get_prompt("query_decomposition").description)
```

## Data Flow

Input: query string, `KnowledgeGraph`, dense embedder, optional sparse embedder and reranker.

Output:

- `SearchEngineRetrieve` from `a_search`
- `SearchEngineResponse` from `a_query`

Used by: applications that need GraphRAG answers, retrieval diagnostics, or mixed retrieval strategies.

## Usage Examples

### Example 1 - Minimal usage

```python
import asyncio

from ragu import BuilderArguments, KnowledgeGraph, NaiveSearchEngine
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI


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
        llm=None,
        embedder=embedder,
        builder_settings=BuilderArguments(build_only_vector_context=True),
    )
    await graph.build_from_docs(["Python is a programming language."])

    engine = NaiveSearchEngine(llm=llm, knowledge_graph=graph, embedder=embedder)
    retrieval = await engine.a_search("What is Python?", top_k=1)
    print(retrieval.result.chunks[0].content)


asyncio.run(main())
```

### Example 2 - Pipeline usage

```python
import asyncio

from ragu import BuilderArguments, GlobalSearchEngine, KnowledgeGraph, LocalSearchEngine, MixSearchEngine
from ragu.graph.types import CommunitySummary, Entity, Relation
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI


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
        builder_settings=BuilderArguments(make_community_summary=False),
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
    await graph.upsert_summaries([CommunitySummary(id="com-1", summary="Python has a creator.")])

    local = LocalSearchEngine(llm=llm, knowledge_graph=graph, embedder=embedder)
    global_ = GlobalSearchEngine(llm=llm, knowledge_graph=graph)
    mix = MixSearchEngine(llm=llm, engines=[local, global_])

    response = await mix.a_query("Summarize the main people and organizations.")
    print(response.response)


asyncio.run(main())
```

### Example 3 - Override a search engine instruction

Every search engine inherits `RaguGenerativeModule`, so `get_prompt` /
`update_prompt` work out of the box:

```python
engine = NaiveSearchEngine(llm=llm, knowledge_graph=graph, embedder=embedder)
engine.get_prompt("naive_search")            # inspect current instruction
engine.update_prompt("naive_search", my_instruction)  # override for this instance only
```

For a full custom-message example see
[`ragu/common/prompts/README.md`](../common/prompts/README.md). Do **not** edit
`DEFAULT_PROMPT_TEMPLATES` directly; `update_prompt` scopes the change to a
single engine instance.

## Integration Points

- LLMs: every `a_query()` renders a prompt and calls `llm.chat_completion`.
- Embedders: local and naive search encode query text through `GraphRetriever`.
- Sparse embedders: local and naive search can issue hybrid dense+sparse vector queries when storage supports sparse vectors.
- Qdrant: hybrid search is handled by vector storage, including Qdrant prefetch/fusion in `QdrantVectorDBStorage`.
- Other modules: search engines read graph artifacts through `KnowledgeGraph.index`.

## Configuration

Shared engine parameters:

- `language`: prompt language, defaulting to `Settings.language`.

Engine token limits are configured centrally via `Settings.llm_context_token_limit`,
`Settings.tokenizer_llm_backend`, and `Settings.tokenizer_llm_name`.

For per-instance control (e.g. several engines with different LLMs / context
windows in the same process), every engine also accepts these constructor
parameters, each defaulting to `None`:

- `max_context_length` (`int | None`) — falls back to `Settings.llm_context_token_limit`.
- `tokenizer_backend` (`Literal["tiktoken", "local"] | None`) — falls back to `Settings.tokenizer_llm_backend`.
- `tokenizer_model` (`str | None`) — falls back to `Settings.tokenizer_llm_name`.

```python
engine = LocalSearchEngine(
    llm=llm, knowledge_graph=kg, embedder=emb,
    max_context_length=16_000,
    tokenizer_model="gpt-4o",
)
```

> Note: an override on `MixSearchEngine` affects **only** its own final-context
> truncation and is **not** propagated to the child engines — each child keeps
> the tokenizer configuration it was constructed with.

Retrieval parameters:

- `top_k`: initial result count for local and naive search.
- `rerank_top_k`: final chunk count for `NaiveSearchEngine` when a reranker is configured.
- `use_summary` and `use_chunks`: toggles for `LocalSearchEngine.a_query`.
- `allow_partial_failures`: controls `MixSearchEngine` child-engine failures.

## Dependencies

Internal:

- `ragu.graph.KnowledgeGraph`
- `ragu.graph.GraphRetriever`
- `ragu.models`
- `ragu.common.prompts`
- `ragu.utils.token_truncation`

External:

- `jinja2`
- `pydantic`
- `typing_extensions`

## Notes / Pitfalls

- `GlobalSearchEngine` needs community summaries; build with `make_community_summary=True` or call `reindex_community()`.
- `LocalSearchEngine` starts from entity vector search, so it needs entity vectors, not just chunk vectors.
- `NaiveSearchEngine` works with `build_only_vector_context=True`.
- `MixSearchEngine` requires at least one child engine.
- `a_search()` is useful for debugging retrieval before involving the LLM.
