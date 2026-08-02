# RAGU examples

Fifteen notebooks covering graph construction, the search engines, the storage
adapters, and the LLM primitives underneath. 

## Setup

Most notebooks need an OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY=...
export LLM_MODEL_NAME=...
export EMBEDDER_MODEL_NAME=...
export OPENAI_BASE_URL=...        # optional, defaults to https://api.openai.com/v1
```

## Notebooks

### Start here

| notebook | shows | needs |
|---|---|---|
| [search_engines_example.ipynb](search_engines_example.ipynb) | Local, global and naive search over one graph, and how to swap an engine's prompt | keys |
| [knowledge_graph_settings_example.ipynb](knowledge_graph_settings_example.ipynb) | Every `KnowledgeGraph` and `BuilderArguments` option, field by field, with the traps | keys (most cells run without) |
| [incremental_updates_example.ipynb](incremental_updates_example.ipynb) | Editing a graph without rebuilding it: adding, forgetting for retention, and fixing extraction errors | **nothing** |

### Search

| notebook | shows | needs |
|---|---|---|
| [mix_search_example.ipynb](mix_search_example.ipynb) | `MixSearchEngine`: ensembling child contexts vs child answers | keys |
| [query_plan_example.ipynb](query_plan_example.ipynb) | `QueryPlanEngine`: decomposing multi-hop questions into a subquery DAG | keys |
| [hybrid_local_search_example.ipynb](hybrid_local_search_example.ipynb) | Dense + BM25 retrieval, when it pays off, and how to tune the built-in stemmer | keys, Qdrant (in-memory by default) |
| [streaming_example.ipynb](streaming_example.ipynb) | `stream_query` on all five engines and what the stream contract guarantees | keys |
| [batch_retrieval_benchmark.ipynb](batch_retrieval_benchmark.ipynb) | 50k vectors, 5k queries: why `asyncio.gather` does not help and batching does | **nothing** |

### Storage

| notebook | shows | needs |
|---|---|---|
| [vector_adapters_example.ipynb](vector_adapters_example.ipynb) | NanoVectorDB vs Qdrant, dense and hybrid, at the `Point` / `EmbeddingHit` level | embedder key |
| [graph_adapters_example.ipynb](graph_adapters_example.ipynb) | The same scenario against NetworkX and Neo4j through one interface | **nothing** (Neo4j cell needs a server) |
| [neo4j_qdrant_setup_example.ipynb](neo4j_qdrant_setup_example.ipynb) | Moving a whole graph onto Neo4j + Qdrant via `StorageArguments` | keys, both servers |

### LLM layer

| notebook | shows | needs |
|---|---|---|
| [llm_embedder_reranker_example.ipynb](llm_embedder_reranker_example.ipynb) | `LLM`, `Embedder` and `Scorer` directly: structured output, streaming, batching | keys + `RERANKER_MODEL_NAME` |
| [custom_prompts_example.ipynb](custom_prompts_example.ipynb) | Finding, reading and replacing any prompt in the registry | keys (discovery cells run without) |
| [llm_as_judge_example.ipynb](llm_as_judge_example.ipynb) | RAGU without a graph — an evaluation harness on the LLM primitives alone | keys |
| [batch_operations_example.ipynb](batch_operations_example.ipynb) | Why nearly every RAGU API is plural: batched calls, retrieval and graph CRUD | keys |

## Running without API keys

`incremental_updates_example` and `batch_retrieval_benchmark` run end to end with no
credentials at all — both use a local stub embedder: the update methods need no LLM, and the benchmark keeps network
latency out of the measurement.
`graph_adapters_example` runs everything except its final Neo4j cell.
`knowledge_graph_settings_example` and `custom_prompts_example` run their
explanatory half: every cell up to the one that builds the API client.

Notebooks needing a server print the `docker run` command they expect.

## Scripts

Older plain-Python examples covering areas the notebooks do not:

| script | shows |
|---|---|
| [extract_with_llm_and_local_search.py](extract_with_llm_and_local_search.py) | Minimal end-to-end build and query |
| [local_embedder_with_short_context.py](local_embedder_with_short_context.py) | Working with a local embedder that has a small context window |
