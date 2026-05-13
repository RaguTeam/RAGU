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
