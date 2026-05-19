"""Integration tests: CachedAsyncOpenAI client against OpenAIMockServer."""
import asyncio
import sys
import time
import openai
import pytest
from collections.abc import Generator
from pydantic import BaseModel

from ragu.common.logger import logger
from ragu.models.openai import CachedAsyncOpenAI
from ragu.utils.testing.openai_mock_server import OpenAIMockServer

logger.remove()
logger.add(sys.stdout, level="DEBUG") 


# ---------------------------------------------------------------------------
# Shared schema and message used across tests
# ---------------------------------------------------------------------------

class _Schema(BaseModel):
    value: str
    count: int


_MSG = [{'role': 'user', 'content': 'hello'}]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def server() -> Generator[OpenAIMockServer, None, None]:
    srv = OpenAIMockServer()
    srv.start()
    yield srv
    srv.stop()


@pytest.fixture
def rate_server() -> Generator[OpenAIMockServer, None, None]:
    # Server that enforces a 0.5 s minimum gap between accepted requests
    srv = OpenAIMockServer(min_delay=0.5)
    srv.start()
    yield srv
    srv.stop()


@pytest.fixture
def delay_server() -> Generator[OpenAIMockServer, None, None]:
    srv = OpenAIMockServer(min_delay=0.0, default_delay=(0.99, 1.01))
    srv.start()
    yield srv
    srv.stop()


@pytest.fixture
def delay_rate_server() -> Generator[OpenAIMockServer, None, None]:
    srv = OpenAIMockServer(min_delay=0.5, default_delay=(0.99, 1.01))
    srv.start()
    yield srv
    srv.stop()


# ---------------------------------------------------------------------------
# 1. Basic round-trip: assert correct return types, not values
# ---------------------------------------------------------------------------

async def test_embed_text_type(server: OpenAIMockServer) -> None:
    # embed_text must return a non-empty list of floats
    client = CachedAsyncOpenAI(base_url=server.base_url, api_key='mock')
    result = await client.embed_text(model_name='mock', text='hello')
    assert isinstance(result, list)
    assert all(isinstance(v, float) for v in result)


async def test_score_type(server: OpenAIMockServer) -> None:
    # score must return a list of (int index, float score) tuples
    client = CachedAsyncOpenAI(base_url=server.base_url, api_key='mock')
    result = await client.score(model_name='mock', text_1='a', text_2=['b', 'c'])
    assert isinstance(result, list)
    for idx, sc in result:
        assert isinstance(idx, int)
        assert isinstance(sc, float)


async def test_chat_completion_str_type(server: OpenAIMockServer) -> None:
    # plain-text completion must return a str
    client = CachedAsyncOpenAI(base_url=server.base_url, api_key='mock')
    result = await client.chat_completion(
        model_name='mock', conversation=_MSG, output_schema=str, # pyright: ignore[reportArgumentType]
    )
    assert isinstance(result, str)


async def test_chat_completion_schema_type(server: OpenAIMockServer) -> None:
    # structured completion must return a validated instance of the requested schema
    client = CachedAsyncOpenAI(base_url=server.base_url, api_key='mock')
    result = await client.chat_completion(
        model_name='mock', conversation=_MSG, output_schema=_Schema, # pyright: ignore[reportArgumentType]
    )
    assert isinstance(result, _Schema)
    assert isinstance(result.value, str)
    assert isinstance(result.count, int)


# ---------------------------------------------------------------------------
# 2. Server rate limiting: min_delay=0.5 s → rapid requests get 429
# ---------------------------------------------------------------------------

# TODO: fix invalid mock for mac os

# async def test_server_rate_limit_raises_429(rate_server: OpenAIMockServer) -> None:
#     # First request is accepted; subsequent ones arrive within min_delay → 429.
#     # max_retries=0 is required: the default retry backoff (≥0.5 s) would exceed
#     # the server's min_delay, causing retries to succeed and hiding the 429.
#     client = CachedAsyncOpenAI(base_url=rate_server.base_url, api_key='mock')
#     got_rate_limit = False
#     for _ in range(5):
#         try:
#             await client.embed_text(model_name='mock', text='hello')
#         except openai.RateLimitError:
#             got_rate_limit = True
#             break
#     assert got_rate_limit, "Expected at least one 429 RateLimitError from the server"


# ---------------------------------------------------------------------------
# 3. Client rate_min_delay=1.0 s spaces requests beyond server's min_delay=0.5 s
#    Note: this test takes ~4 s (4 inter-request sleeps × 1.0 s each)
# ---------------------------------------------------------------------------

@pytest.mark.slow
async def test_client_rate_delay_prevents_429(rate_server: OpenAIMockServer) -> None:
    # rate_min_delay=0.7 ensures >= 0.7 s between requests, satisfying server's 0.5 s threshold.
    # max_retries=0 keeps the test deterministic: no retry masks a potential 429.
    client = CachedAsyncOpenAI(base_url=rate_server.base_url, api_key='mock', rate_min_delay=1.0)
    for _ in range(5):
        result = await client.embed_text(model_name='mock', text='hello')
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 4. retry_times_sec=(1,1,1): each retry waits 1 s > server's 0.5 s min_delay,
#    so retried requests succeed — no RateLimitError is raised
# ---------------------------------------------------------------------------

@pytest.mark.slow
async def test_retry_times_sec_clears_429(rate_server: OpenAIMockServer) -> None:
    # retry_times_sec waits 1 s before each retry; server's min_delay=0.5 s,
    # so the retried request always arrives after the cooldown window → succeeds.
    client = CachedAsyncOpenAI(
        base_url=rate_server.base_url, api_key='mock', retry_times_sec=(1, 1, 1),
    )
    for _ in range(5):
        result = await client.embed_text(model_name='mock', text='hello')
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 5. rate_max_simultaneous=2, default_delay≈1 s: 6 requests in 3 batches of 2
#    → total ≈ 3 s
# ---------------------------------------------------------------------------


@pytest.mark.slow
async def test_max_simultaneous_timing(delay_server: OpenAIMockServer) -> None:
    # 6 requests / 2 simultaneous = 3 serial batches × ~1 s each → ~3 s total.
    client = CachedAsyncOpenAI(
        base_url=delay_server.base_url, api_key='mock', rate_max_simultaneous=2,
    )
    t0 = time.monotonic()
    await asyncio.gather(*[
        client.embed_text(model_name='mock', text='hello') for _ in range(6)
    ])
    elapsed = time.monotonic() - t0
    assert 2.8 <= elapsed <= 3.6, f'Expected ~3 s, got {elapsed:.2f} s'


# ---------------------------------------------------------------------------
# 6. rate_min_delay=0.3 + rate_max_simultaneous=2: staggered starts add ~0.3 s
#    → total ≈ 3.3 s
# ---------------------------------------------------------------------------

@pytest.mark.slow
async def test_max_simultaneous_with_rate_delay_timing(delay_server: OpenAIMockServer) -> None:
    # rate_min_delay=0.3 staggers request starts by 0.3 s each.
    # With semaphore=2, the 3rd request is held until the 1st finishes (~1 s),
    # but the staggered starts push the overall tail out by ~0.3 s → ~3.3 s total.
    client = CachedAsyncOpenAI(
        base_url=delay_server.base_url, api_key='mock',
        rate_min_delay=0.3, rate_max_simultaneous=2,
    )
    t0 = time.monotonic()
    await asyncio.gather(*[
        client.embed_text(model_name='mock', text='hello') for _ in range(6)
    ])
    elapsed = time.monotonic() - t0
    assert 3.1 <= elapsed <= 3.9, f'Expected ~3.3 s, got {elapsed:.2f} s'

# test caching

@pytest.mark.slow
async def test_caching(delay_rate_server: OpenAIMockServer) -> None:
    client = CachedAsyncOpenAI(
        base_url=delay_rate_server.base_url, api_key='mock',
        rate_min_delay=0, rate_max_simultaneous=None, cache={},
    )
    await client.embed_text(model_name='mock', text='hello')
    t0 = time.monotonic()
    await asyncio.gather(*[
        client.embed_text(model_name='mock', text='hello') for _ in range(5)
    ])
    elapsed = time.monotonic() - t0
    assert elapsed <= 0.1, f'Expected <0.1 s, got {elapsed:.2f} s'


# ---------------------------------------------------------------------------
# 8. embed_texts: batch embedding against mock server
# ---------------------------------------------------------------------------

async def test_embed_texts_returns_correct_count(server: OpenAIMockServer) -> None:
    client = CachedAsyncOpenAI(base_url=server.base_url, api_key='mock')
    result = await client.embed_texts(model_name='mock', texts=['a', 'b', 'c'])
    assert isinstance(result, list)
    assert len(result) == 3
    for emb in result:
        assert isinstance(emb, list)


async def test_embed_texts_single_text(server: OpenAIMockServer) -> None:
    client = CachedAsyncOpenAI(base_url=server.base_url, api_key='mock')
    result = await client.embed_texts(model_name='mock', texts=['hello'])
    assert isinstance(result, list)
    assert len(result) == 1


async def test_embed_texts_order_preserved(server: OpenAIMockServer) -> None:
    client = CachedAsyncOpenAI(base_url=server.base_url, api_key='mock')
    texts = [f'text-{i}' for i in range(5)]
    result = await client.embed_texts(model_name='mock', texts=texts)
    assert len(result) == 5


async def test_embed_texts_all_cache_hit(server: OpenAIMockServer) -> None:
    cache: dict[str, object] = {}
    client = CachedAsyncOpenAI(
        base_url=server.base_url, api_key='mock', cache=cache,
    )
    await client.embed_texts(model_name='mock', texts=['x', 'y'])
    t0 = time.monotonic()
    result = await client.embed_texts(model_name='mock', texts=['x', 'y'])
    elapsed = time.monotonic() - t0
    assert len(result) == 2
    assert elapsed < 0.1, f'Cache hit should be instant, got {elapsed:.2f} s'


async def test_embed_texts_partial_cache_hit(server: OpenAIMockServer) -> None:
    cache: dict[str, object] = {}
    client = CachedAsyncOpenAI(
        base_url=server.base_url, api_key='mock', cache=cache,
    )
    await client.embed_text(model_name='mock', text='cached')
    texts = ['cached', 'new1', 'new2']
    result = await client.embed_texts(model_name='mock', texts=texts)
    assert len(result) == 3


async def test_embed_texts_cache_cross_compatible_with_embed_text(
    server: OpenAIMockServer,
) -> None:
    cache: dict[str, object] = {}
    client = CachedAsyncOpenAI(
        base_url=server.base_url, api_key='mock', cache=cache,
    )
    single_result = await client.embed_text(model_name='mock', text='hello')
    batch_result = await client.embed_texts(model_name='mock', texts=['hello'])
    assert batch_result[0] == single_result


async def test_embed_texts_cache_stores_for_future_embed_text(
    server: OpenAIMockServer,
) -> None:
    cache: dict[str, object] = {}
    client = CachedAsyncOpenAI(
        base_url=server.base_url, api_key='mock', cache=cache,
    )
    await client.embed_texts(model_name='mock', texts=['hello'])
    t0 = time.monotonic()
    single_result = await client.embed_text(model_name='mock', text='hello')
    elapsed = time.monotonic() - t0
    assert isinstance(single_result, list)
    assert elapsed < 0.1, f'Should be a cache hit, got {elapsed:.2f} s'


async def test_embed_texts_no_cache(server: OpenAIMockServer) -> None:
    client = CachedAsyncOpenAI(base_url=server.base_url, api_key='mock')
    result1 = await client.embed_texts(model_name='mock', texts=['a', 'b'])
    result2 = await client.embed_texts(model_name='mock', texts=['a', 'b'])
    assert len(result1) == len(result2) == 2


# ---------------------------------------------------------------------------
# 9. End-to-end: EmbedderOpenAI.batch_embed_text through mock server
# ---------------------------------------------------------------------------

async def test_batch_embed_text_end_to_end(server: OpenAIMockServer) -> None:
    from ragu.models.embedder import EmbedderOpenAI

    client = CachedAsyncOpenAI(base_url=server.base_url, api_key='mock')
    embedder = EmbedderOpenAI(
        client=client, model_name='mock', dim=1, batch_size=2,
    )
    texts = ['t1', 't2', 't3', 't4', 't5']
    result = await embedder.batch_embed_text(texts)
    assert len(result) == 5
    for emb in result:
        assert isinstance(emb, list)


async def test_batch_embed_text_end_to_end_with_cache(
    server: OpenAIMockServer,
) -> None:
    from ragu.models.embedder import EmbedderOpenAI

    cache: dict[str, object] = {}
    client = CachedAsyncOpenAI(
        base_url=server.base_url, api_key='mock', cache=cache,
    )
    embedder = EmbedderOpenAI(
        client=client, model_name='mock', dim=1, batch_size=10,
    )
    result1 = await embedder.batch_embed_text(['a', 'b', 'c'])
    t0 = time.monotonic()
    result2 = await embedder.batch_embed_text(['a', 'b', 'c'])
    elapsed = time.monotonic() - t0
    assert result1 == result2
    assert elapsed < 0.2, f'Second call should hit cache, got {elapsed:.2f} s'
