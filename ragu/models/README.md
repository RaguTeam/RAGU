# Module: ragu.models

## Role in RAGU Pipeline

`ragu.models` provides the model abstractions and network layer used by every
LLM-driven stage of RAGU: chat completion (extraction, summarization, answering)
and embedding (vector indexing and retrieval). It also ships sparse embedders
for hybrid search and scorers for reranking.

Pipeline position:

```text
prompt/Chunk/query -> LLM / Embedder / SparseEmbedder / Scorer -> CachedAsyncOpenAI -> API
```

## Overview

The module separates **interface level** (`LLM`, `Embedder`, `Scorer`,
`SparseEmbedder`) from **network level** (`CachedAsyncOpenAI`). Interfaces are
the argument types consumed by RAGU components; the network client handles
caching, retries, rate limiting, and the OpenAI-compatible HTTP calls. A single
`CachedAsyncOpenAI` instance can be shared between an `LLMOpenAI` and an
`EmbedderOpenAI`, or they can use separate clients (see Configuration).

## Key Components

### LLM and LLMOpenAI

`LLM` is the abstract chat-completion interface used by extractors,
summarizers, and search engines. `LLMOpenAI` is the OpenAI-compatible
implementation.

- Purpose: render prompts to OpenAI messages and return text or Pydantic
  structured output.
- Important methods: `chat_completion`, `batch_chat_completion`.
- Important parameters: `client` (`CachedAsyncOpenAI`), `model_name`.

```python
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI

client = CachedAsyncOpenAI(
    base_url="https://api.openai.com/v1",
    api_key="dummy-api-token",
)
llm = LLMOpenAI(client=client, model_name="gpt-4o-mini")
```

### Embedder and EmbedderOpenAI

`Embedder` is the abstract embedding interface. `EmbedderOpenAI` is the
OpenAI-compatible implementation.

- Purpose: produce dense vectors for entities, relations, chunks, and queries.
- All input texts are **automatically truncated** to the token limit before the
  API call (see [`ragu/common/README.md`](../common/README.md)).
- Important parameters of `EmbedderOpenAI`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `client` | `CachedAsyncOpenAI` instance (required) | — |
| `model_name` | Embedding model ID (required) | — |
| `dim` | Embedding dimension; auto-detected via `initialize()` if omitted | `None` |
| `batch_size` | Max texts per single API call | `500` |
| `max_concurrent_batches` | Max concurrent batch API calls | `5` |
| `embedder_token_limit` | Max tokens per input text (overrides `Settings`) | `None` |
| `tokenizer_backend` | `"tiktoken"` or `"local"` (overrides `Settings`) | `None` |
| `tokenizer_name` | Tokenizer model ID (overrides `Settings`) | `None` |

```python
from ragu.models.embedder import EmbedderOpenAI

embedder = EmbedderOpenAI(
    client=client,
    model_name="text-embedding-3-large",
    dim=3072,
)
await embedder.initialize()  # optional; auto-detects dim if not set
```

### CachedAsyncOpenAI

Network-level client shared by `LLMOpenAI` and `EmbedderOpenAI`. Controls
caching, retries, rate limiting, timeouts, and an optional debug store for
failed requests. This is the canonical reference for client configuration; the
main README and the docs link here instead of duplicating it.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `base_url`, `api_key` | OpenAI-compatible endpoint credentials | — |
| `rate_max_simultaneous` | Max concurrent in-flight requests | `None` (unlimited) |
| `rate_max_per_minute` | Max requests per minute | `None` (unlimited) |
| `rate_min_delay` | Min seconds between request starts | `None` |
| `retry_times_sec` | Retry wait schedule on transient errors | `(2, 4, 8)` |
| `embed_timeout` | Per-request timeout for embedding calls (seconds) | `60.0` |
| `cache` | Cache directory path. When `None`, falls back to `Settings.cache_path` (also `None` by default → caching disabled). An explicit value (incl. in-memory `{}`) always takes precedence. | `None` |
| `debug_errors_storage` | Directory for dumping failed request payloads. When `None`, falls back to `Settings.debug_errors_path` (also `None` by default). | `None` |

> **Tip:** instead of passing `cache=` to every client, set a process-wide
> default once via `Settings.cache_path = "./my_cache"` (and similarly
> `Settings.debug_errors_path`). The cache path must be a **stable, long-lived
> directory independent of `Settings.storage_folder`** (which is per-run). See
> `ragu/common/README.md` for the invalidation caveats.

#### Shared client (simpler, works for most cases)

```python
client = CachedAsyncOpenAI(
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    rate_max_simultaneous=10,
    rate_max_per_minute=100,
    cache="./llm_cache",
)

llm = LLMOpenAI(client=client, model_name="gpt-4o-mini")
embedder = EmbedderOpenAI(client=client, model_name="text-embedding-3-large", dim=3072)
```

Rate limiting is shared between all models that use the same client instance.
This is convenient for small-to-medium workloads.

#### Separate clients for LLM and embedder (large corpora)

When processing large corpora (thousands of entities and relations), LLM calls
(slow, seconds per request) and embedding calls (fast but numerous) compete for
the same connection pool and rate limiter. Using separate clients isolates
their resources.

```python
llm_client = CachedAsyncOpenAI(
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    rate_max_simultaneous=5,
    rate_max_per_minute=60,
    retry_times_sec=(4, 8, 16),
    cache="./llm_cache",
)

embed_client = CachedAsyncOpenAI(
    base_url="https://api.openai.com/v1",  # can be a different provider
    api_key="sk-...",
    rate_max_simultaneous=20,
    rate_max_per_minute=500,
    embed_timeout=60.0,
    cache="./embed_cache",
)

llm = LLMOpenAI(client=llm_client, model_name="gpt-4o-mini")
embedder = EmbedderOpenAI(
    client=embed_client,
    model_name="text-embedding-3-large",
    dim=3072,
    batch_size=500,
    max_concurrent_batches=5,
)
```

**When a shared client is sufficient:**

- Small to medium documents (up to ~1000 entities/relations).
- Generous API provider rate limits.
- LLM and embedder on the same endpoint.

**When separate clients are recommended:**

- Large corpora (thousands of entities and relations).
- Strict API rate limits (low RPM).
- LLM and embedder on different providers or endpoints.
- You need independent scaling of LLM vs. embedding throughput.

### Batch embedding

`EmbedderOpenAI` uses API-level batching by default: texts are grouped into
sub-batches (default 500 texts per batch) and sent to the `/embeddings` endpoint
as `input=[t1, t2, ...]` in a single HTTP request. This dramatically reduces the
number of HTTP requests compared to sending one text per request. A semaphore
(`max_concurrent_batches`, default 5) limits the number of concurrent batch API
calls, preventing connection-pool exhaustion.

### Sparse embedders (BM25, BM42, SPLADE)

`SparseEmbedder` is the abstract sparse-vector interface. Concrete
implementations are `BM25` (FastEmbed BM25, no API calls), `BM42` (FastEmbed
BM42), and `SPLADE`. Sparse embeddings power hybrid retrieval in
`QdrantVectorDBStorage`. See [`ragu/storage/vdb_storage_adapters/README.md`](../storage/vdb_storage_adapters/README.md)
for pairing sparse embedders with a hybrid vector backend.

### Scorer (rerankers)

`Scorer` is the abstract reranker interface. Implementations are `ScorerOpenAI`
(Cohere/OpenAI-compatible reranker API) and `ScorerCrossEncoder`
(local `sentence_transformers.CrossEncoder`). Scorers are optional and used by
search engines to rerank retrieved chunks.

## Data Flow

Input: prompt messages (LLM), text or batched texts (Embedder/SparseEmbedder),
query+documents pairs (Scorer).

Output: chat completions (text or Pydantic objects), dense vectors, sparse
vectors, relevance scores.

Used by:

- `ragu.triplet` extractors (LLM)
- `ragu.graph` summarizers and community detection (LLM, Embedder)
- `ragu.graph.Index` (Embedder, optional SparseEmbedder)
- `ragu.search_engine` (LLM, Embedder, optional Scorer)

## Integration Points

- `ChatMessages.to_openai()` produces the payloads consumed by `LLMOpenAI`.
- `Index` embeds entity/relation/chunk text before vector upsert.
- `GraphRetriever` uses the embedder to build query vectors.
- Sparse embedders must be paired with a vector backend configured for sparse
  vectors.

## Configuration

Client and rate-limiting parameters are documented in the `CachedAsyncOpenAI`
table above (the canonical reference). Token-limit and tokenizer defaults are
centralized in `Settings` — see [`ragu/common/README.md`](../common/README.md).

## Dependencies

Internal:

- `ragu.common.logger`
- `ragu.common.global_parameters.Settings`
- `ragu.utils.token_truncation.TokenTruncation`

External:

- OpenAI Python SDK (`openai`)
- `tenacity` (transient-error retries)
- `diskcache` (response cache and debug store)
- `tiktoken` (optional, for `"tiktoken"` tokenizer backend)
- `fastembed` (sparse embedders `BM25`, `BM42`)
- `sentence_transformers` (optional, `ScorerCrossEncoder` and local tokenizers)

## Debugging

Enable debug logging and dump failed requests to disk:

```python
import os
import sys

from ragu.common.logger import logger
from ragu.models.openai import CachedAsyncOpenAI

logger.remove()
logger.add(sys.stdout, level="DEBUG")

client = CachedAsyncOpenAI(
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
    rate_min_delay=2,
    rate_max_simultaneous=10,
    retry_times_sec=(4, 8),
    cache="./llm_cache",
    debug_errors_storage="./llm_debug",
)
```

On error you can open the corresponding request:

```python
from diskcache import Index

index = Index("./llm_debug")
info = index["1772078734557794672"]
print(list(info["kwargs"]))
print(info["kwargs"]["output_schema"].model_json_schema())
print(info["kwargs"]["conversation"][0]["content"])
```

And re-run it:

```python
result = await client._uncached_raw_chat_completion(
    **info["kwargs"],
)
```

## Notes / Pitfalls

- Rate limiting is **per client instance**. Models sharing a client share the
  limiter; models on separate clients are isolated.
- `CachedAsyncOpenAI` sets `max_retries=0` on the underlying `AsyncOpenAI` and
  handles retries itself via `tenacity` and `retry_times_sec`.
- `EmbedderOpenAI` truncates input before embedding; a too-small
  `embedder_token_limit` silently changes vector quality.
- `embed_timeout` only applies to embedding requests, not chat requests.
- For local GPU serving (e.g. vLLM) lower `batch_size` and
  `max_concurrent_batches` to avoid OOM — the defaults (500 / 5) target cloud
  APIs.
