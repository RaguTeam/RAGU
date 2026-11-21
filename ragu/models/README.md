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

- Purpose: render prompts to OpenAI messages and return text, Pydantic
  structured output, or streamed text deltas.
- Important methods: `chat_completion`, `batch_chat_completion`,
  `stream_chat_completion`.
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

Streaming is text-only and bypasses response caching. Use regular
`chat_completion` / `batch_chat_completion` for structured Pydantic output.

```python
async for delta in llm.stream_chat_completion([
    {"role": "user", "content": "Answer briefly: what is RAG?"},
]):
    print(delta, end="")
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
caching, retries, rate limiting, timeouts, streaming chat completions, and an
optional debug store for failed requests. This is the canonical reference for
client configuration; the main README and the docs link here instead of
duplicating it.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `base_url`, `api_key` | OpenAI-compatible endpoint credentials | — |
| `rate_max_simultaneous` | Max concurrent in-flight requests | `None` (unlimited) |
| `rate_max_per_minute` | Max requests per minute | `None` (unlimited) |
| `rate_min_delay` | Min seconds between request starts | `None` |
| `retry_times_sec` | Retry wait schedule on transient errors | `(2, 4, 8)` |
| `timeout` | Shared per-request timeout for chat (including structured output and streaming), embeddings, Decisions and scoring. `None` preserves the configured SDK timeout. | `60.0` effective |
| `embed_timeout` | Backward-compatible alias for the same shared timeout; its positional argument is preserved. Explicit `timeout` takes precedence, including `timeout=None`. | `60.0` |
| `cache` | Cache directory path. When `None`, falls back to `Settings.cache_path` (also `None` by default → caching disabled). An explicit value (incl. in-memory `{}`) always takes precedence. | `None` |
| `debug_errors_storage` | Directory for dumping failed request payloads. When `None`, falls back to `Settings.debug_errors_path` (also `None` by default). | `None` |

Use `CachedAsyncOpenAI(timeout=30.0, ...)` for a common limit. Existing calls
such as `CachedAsyncOpenAI(embed_timeout=30.0, ...)` remain valid and now set
the same common limit for all operations. If both names are supplied,
`timeout` wins. Omitting both retains the effective 60-second default.
`timeout=None` (or `embed_timeout=None` when `timeout` is omitted) inherits the
configured SDK timeout. Per-operation timeout parameters are unnecessary.

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
    timeout=60.0,
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

## Decisions: typed questions about one state

`DecisionClient` is a structural `Protocol` with one async operation:
`decide(model_name, *, state, questions) -> DecisionResult`. Both
`CachedAsyncOpenAI` and `TypeSafeClient` implement it. Select and configure the
transport in application assembly; one instance accepts multiple model names.
This API evaluates one state. It does not implement the RAGU `Scorer` interface.

```python
import asyncio
import os

from ragu import (
    DecisionClient, PredicateAnswer, PredicateQuestion, RefusalAnswer,
    TypeSafeClient,
)


async def check_claims(client: DecisionClient, model_name: str) -> None:
    result = await client.decide(
        model_name=model_name,
        state={
            "claim_a": "The company had exactly one branch on January 1, 2025.",
            "claim_b": "The same company had three branches on January 1, 2025.",
        },
        questions=[PredicateQuestion(
            name="contradiction",
            instructions="Do the claims contradict each other under these conditions? "
                         "A contradiction means they cannot both be true at once.",
        )],
    )
    answer = result.answers["contradiction"]
    if isinstance(answer, PredicateAnswer):
        print(f"Contradiction probability: {answer.probability:.3f}")
    elif isinstance(answer, RefusalAnswer):
        print("The model declined to answer.")


async def main() -> None:
    async with TypeSafeClient(
        api_key=os.environ["OPENROUTER_API_KEY"],
        url="https://openrouter.ai/api/alpha/decisions",
    ) as client:
        await check_claims(client, "typesafe/jev-1.14")


if __name__ == "__main__":
    asyncio.run(main())
```

For direct TypeSafe, construct `TypeSafeClient(os.environ["TYPESAFE_API_KEY"])`
and pass an available Jev model to `check_claims`. For OpenAI, pass an existing
`CachedAsyncOpenAI` to the same function and use a supported Decisions model.
The operation uses `AsyncOpenAI.decisions.create` from `openai>=3.57.0,<4`,
through the configured SDK instance. Custom base URL prefixes, project,
organization, headers and connection pool are preserved. SDK retries are
turned off for this operation using `with_options(max_retries=0)`, without
changing the injected instance. Chat/embedding parameters are not forwarded.

### Questions and answers

`state` accepts a string, JSON object or JSON array. Nested values accept JSON
scalars including `None`; object keys must be strings and numbers must be
finite. Tuples, arbitrary objects and cyclic structures are rejected.
`questions` is a nonempty ordered sequence with unique, nonempty names and text
instructions. All question types can be mixed in one request.

| Public question types | Fields beyond `name` and `instructions` | Public answer types |
|---|---|---|
| `PredicateQuestion` | — | `PredicateAnswer(probability)` in [0, 1] |
| `ChoiceQuestion` | `choices`: ordered `ChoiceOption(value, description)` objects with unique string values | `ChoiceAnswer(choice, probabilities, confidence)`; probabilities keyed by original values |
| `ScoreQuestion` | `levels`: 2–10 ordered `ScoreLevel(label, description)` objects, lowest first | `ScoreAnswer(score, probabilities, confidence)`; probabilities keyed by integer indices |
| Any question | Explicit service refusal | `RefusalAnswer(details=None)`; other answers remain available |

Names, instructions, choice values/descriptions and level labels/descriptions
must contain non-whitespace text. OpenAI accepts 1–400 questions and 2–255
choices per choice question. No additional undocumented Jev limits are imposed.
A score for K levels stays on its original 0…K−1 scale and may be fractional.
Confidence is the provider's independent [0, 1] value; it is not recomputed.
Distributions must cover exactly the supplied choices/levels. RAGU allows an
absolute probability-sum error of **0.01** for rounding; every probability
must still be finite and within [0, 1]. Zero is preserved.

`DecisionQuestion`, `DecisionAnswer` and `DecisionState` are public type aliases.
`DecisionResult.answers` maps original names to typed answers in input order.
The result also exposes `requested_model`, `resolved_model`, `provider` (only
when supplied by the service), `id`, `usage`, and `cache_hit`. Missing metadata
is `None`. `usage` preserves the provider's token counts, cost and additional
usage fields as a dictionary. Cached usage describes the original call.

### TypeSafeClient configuration and lifecycle

| Parameter | Default | Meaning |
|---|---|---|
| `api_key` | Required | Caller-supplied Bearer token, regardless of host |
| `url` | `https://api.typesafe.ai/v1/systemone` | Full absolute HTTP(S) POST URL, including path and optional query; no suffix is appended |
| `http_client` | `None` | Optional caller-owned `httpx.AsyncClient`; otherwise one pool is created |
| `timeout` | `60.0` | HTTPX timeout in seconds for every attempt |
| `cache` | `None` | Mapping or disk path; otherwise `Settings.cache_path` |
| `cache_prefix` | `"typesafe"` | Cache namespace |
| `rate_min_delay` | `None` | Minimum interval between request starts, seconds |
| `rate_max_per_minute` | `None` | Maximum request starts per minute |
| `rate_max_simultaneous` | `None` | Concurrent network attempt limit |
| `retry_times_sec` | `(2, 4, 8)` | Retry delays; `None` or empty means one attempt |

TypeSafe and OpenRouter use the same Jev codec. OpenRouter uses
`https://openrouter.ai/api/alpha/decisions` in this integration; its
`/api/v1/systemone` compatibility route is also documented upstream. A proxy
can use any full operation URL and receives the same request body. Host names
do not select credentials or protocols.

Both concrete clients support `async with` and idempotent `aclose()`. They close
only SDK/HTTP clients they created. An injected client remains owned by the
application and usable after exiting the wrapper context.

### Caching, retries and errors

Decision caching uses `ResponseCachingMixin`. Explicit mappings (including
`{}`) or disk paths take precedence over `Settings.cache_path`. Only complete,
validated results are stored, serialized as JSON. Cache keys hash the format
version, namespace, protocol, full operation URL, model and full request,
including meaningful question, choice and level order. Authorization keys are
excluded. Cache hits bypass all network and rate-limit work and return
`cache_hit=True`. Use a new `cache_prefix` or a pinned model version when a
model alias changes during an experiment.

OpenAI decisions share the existing client's quota with chat and embeddings
and use RAGU's tenacity retry classification and schedule. Jev retries timeouts,
transient network failures and HTTP 429/500/502/503/504. A valid `Retry-After`
number or HTTP date is a minimum retry delay. Each attempt reacquires the
instance's limiters; the semaphore is released before retry sleep. Task
cancellation propagates to the caller and frees its slot.

Input and response validation raise `ValueError` naming the offending field;
protocol errors also identify the provider and question when available.
Missing, duplicate, unexpected or mismatched answers fail the whole call and
are not cached. An explicit refusal is a valid individual answer. Authentication
errors, bad requests, invalid JSON and malformed answers are not retried. HTTP
exceptions retain their status (`openai.APIStatusError` or
`httpx.HTTPStatusError`). Standard decision logs omit state and credentials.
`CachedAsyncOpenAI.debug_errors_storage` retains its existing **opt-in**
contract: when enabled, failing decision arguments include the supplied state.

Offline tests and separate provider fixtures live in `tests/decisions/`.
Live smoke tests require `RAGU_DECISIONS_SMOKE=1`, the route's API key and its
explicit model environment variable (`TYPESAFE_DECISION_MODEL`,
`OPENROUTER_DECISION_MODEL`, or `OPENAI_DECISION_MODEL`). Run them with
`pytest tests/decisions/test_smoke.py -m integration`; default CI needs no keys.

For the OpenRouter mixed-request and cache check, put `OPENROUTER_API_KEY` in
`.env` at the repository root and run:

```bash
RAGU_DECISIONS_SMOKE=1 pytest tests/decisions/test_openrouter_live.py -m integration --no-cov -q -s
```

This opt-in test uses `typesafe/jev-1.14` unless `OPENROUTER_DECISION_MODEL` is
set in `.env` or the environment. It sends Russian claims with nested JSON,
asks all three question types, validates the typed answers and their ranges,
and repeats the request to verify a local cache hit with one HTTP POST in total.
Successful output includes answers, resolved model, provider and usage, without
the API key. HTTP attempts use a 15-second timeout and retries are disabled.

Protocol references: [TypeSafe System One](https://docs.typesafe.ai/api/system-one),
[OpenRouter decisions](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request),
[OpenAI Decisions SDK](https://github.com/openai/openai-python/blob/v3.57.0/src/openai/resources/decisions.py).

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
from ragu.utils.ragu_utils import get_disk_cache

index = get_disk_cache("./llm_debug")
info = index["1772078734557794672"]
print(list(info["kwargs"]))
print(info["kwargs"]["output_schema"])
print(info["kwargs"]["conversation"][0]["content"])
```

And re-run it, passing the schema class again:

```python
result = await client._uncached_raw_chat_completion(
    **{**info["kwargs"], "output_schema": MySchema},
)
```

## Notes / Pitfalls

- Rate limiting is **per client instance**. Models sharing a client share the
  limiter; models on separate clients are isolated.
- `CachedAsyncOpenAI` sets `max_retries=0` on the underlying `AsyncOpenAI` and
  handles retries itself via `tenacity` and `retry_times_sec`.
- `EmbedderOpenAI` truncates input before embedding; a too-small
  `embedder_token_limit` silently changes vector quality.
- `timeout` applies to every request made by `CachedAsyncOpenAI`. The legacy
  `embed_timeout` name reads and writes the same setting. When `timeout=None`,
  SDK operations inherit the injected SDK client's timeout; the HTTP scoring
  request uses that same timeout. The injected SDK instance is not modified.
- For local GPU serving (e.g. vLLM) lower `batch_size` and
  `max_concurrent_batches` to avoid OOM — the defaults (500 / 5) target cloud
  APIs.
