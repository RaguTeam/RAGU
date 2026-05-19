# Interface level

`LLM`, `Embedder` - interface level. Argument type for
various components. Example:

```
@dataclass
class MyCommunitySummarizer:
    fast_llm: LLM,
    heavy_llm: LLM,
    query_embedder: Embedder
    key_embedder: Embedder
```

# Network level

`ResponseCachingMixin`, `CachedAsyncOpenAI` work on network level.
They handle caching, retrying, rate limiting, and working with
OpenAI API.

A single `CachedAsyncOpenAI` can be shared between LLM and embedder:

```
client = CachedAsyncOpenAI()
summarizer = MyCommunitySummarizer(
    fast_llm=LLMOpenAI(client, 'qweno-tiny'),
    heavy_llm=LLMOpenAI(client, 'claude-mighty'),
    query_embedder=EmbedderOpenAI(client, 'bert', dim=768),
    key_embedder=EmbedderOpenAI(client, 'bert', dim=768),
)
```

Rate limiting is shared between all models that use the same client
instance. This is convenient for small-to-medium workloads.

## Separate clients for LLM and embedder

For large corpora (thousands of entities/relations), LLM calls (slow,
seconds per request) and embedding calls (fast but numerous) compete
for the same connection pool and rate limiter.  Use separate clients
to isolate their resources:

```
llm_client = CachedAsyncOpenAI(
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    rate_max_simultaneous=5,
    rate_max_per_minute=60,
    retry_times_sec=(4, 8, 16),
)

embed_client = CachedAsyncOpenAI(
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    rate_max_simultaneous=20,
    rate_max_per_minute=500,
    embed_timeout=60.0,
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

## Batch embedding

`EmbedderOpenAI` uses API-level batching by default: texts are grouped
into sub-batches (default 500 texts per batch) and sent to the
`/embeddings` endpoint as `input=[t1, t2, ...]` in a single HTTP
request. This dramatically reduces the number of HTTP requests compared
to sending one text per request.

A semaphore (`max_concurrent_batches`, default 5) limits the number of
concurrent batch API calls, preventing connection-pool exhaustion.

# Example debugging

```
from ragu.common.logger import logger
logger.remove()
logger.add(sys.stdout, level="DEBUG") 

client = CachedAsyncOpenAI(
    base_url=os.environ['OPENAI_BASE_URL'],
    api_key=os.environ['OPENAI_API_KEY'],
    rate_min_delay=2,
    rate_max_simultaneous=10,
    retry_times_sec=(4, 8),
    cache='./llm_cache',
    debug_errors_storage='./llm_debug',
)
```

On error you can open the corresponding request:

```
from diskcache import Index
index = Index('./llm_debug')
info = index['1772078734557794672']
print(list(info['kwargs']))
print(info['kwargs']['output_schema'].model_json_schema())
print(info['kwargs']['conversation'][0]['content'])
```

And re-run it:

```
result = await client._uncached_raw_chat_completion(
    **info['kwargs'],
)
```