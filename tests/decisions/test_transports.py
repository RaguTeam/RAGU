import asyncio
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import openai
import pytest

from ragu import CachedAsyncOpenAI, PredicateQuestion, ScoreAnswer, TypeSafeClient
from ragu.common.global_parameters import Settings
from ragu.models.typesafe import _retry_after


FIXTURES = Path(__file__).parent / 'fixtures'


def fixture(name):
    return json.loads((FIXTURES / f'{name}.json').read_text())


@pytest.mark.parametrize('url,source', [
    ('https://api.typesafe.ai/v1/systemone', 'typesafe'),
    ('https://openrouter.ai/api/alpha/decisions', 'openrouter'),
    ('https://proxy.example/prefix/op?version=1', 'typesafe'),
])
async def test_jev_post_and_ownership(url, source, questions):
    calls = []

    async def handle(request):
        calls.append(request)
        assert str(request.url) == url
        assert request.headers['authorization'] == 'Bearer secret'
        body = json.loads(request.content)
        assert body['state'] == {'русский': [1, None]}
        assert list(body['questions']) == ['predicate', 'choice', 'score']
        return httpx.Response(200, json=fixture(source))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as pool:
        async with TypeSafeClient('secret', url=url, http_client=pool, cache={}) as client:
            first = await client.decide('m', state={'русский': [1, None]}, questions=questions)
            second = await client.decide('m', state={'русский': [1, None]}, questions=questions)
            assert not first.cache_hit and second.cache_hit
            assert first.answers == second.answers and first.usage == second.usage
            assert isinstance(second.answers['score'], ScoreAnswer)
            assert list(second.answers['score'].probabilities) == [0, 1, 2]
            assert len(calls) == 1
        await client.aclose()
        assert not pool.is_closed


@pytest.mark.parametrize('first,attempts', [(429, 2), (500, 2), (502, 2), (503, 2), (504, 2), ('timeout', 2), ('connect', 2), (401, 1), (400, 1), ('json', 1), ('invalid', 1)])
async def test_jev_retry_policy(first, attempts, questions):
    count = 0
    cache = {}

    async def handle(request):
        nonlocal count
        count += 1
        if count > 1:
            return httpx.Response(200, json=fixture('typesafe'))
        if first == 'timeout':
            raise httpx.ReadTimeout('test', request=request)
        if first == 'connect':
            raise httpx.ConnectError('test', request=request)
        if first == 'json':
            return httpx.Response(200, content=b'{broken')
        if first == 'invalid':
            return httpx.Response(200, json={'answers': {}})
        return httpx.Response(first, json={'error': 'test'})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as pool:
        client = TypeSafeClient('key', http_client=pool, retry_times_sec=(0, 0, 0), cache=cache)
        if attempts == 1:
            with pytest.raises((httpx.HTTPStatusError, ValueError)) as error:
                await client.decide('m', state='x', questions=questions)
            if first in (401, 400):
                assert error.value.response.status_code == first
            assert not cache
        else:
            await client.decide('m', state='x', questions=questions)
        assert count == attempts


@pytest.mark.parametrize('schedule,total', [(None, 1), ((), 1), ((0, 0, 0), 4)])
async def test_jev_retry_exhaustion(schedule, total, questions):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(429, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as pool:
        client = TypeSafeClient('key', http_client=pool, retry_times_sec=schedule)
        with pytest.raises(httpx.HTTPStatusError):
            await client.decide('m', state='x', questions=questions)
    assert len(calls) == total


async def test_jev_retry_after_releases_slot(monkeypatch, questions):
    calls = 0
    waiting, release = asyncio.Event(), asyncio.Event()
    client = None
    waits = []

    async def sleep(delay):
        waits.append(delay)
        assert not client._rate_limit_contexts[0].locked()
        waiting.set()
        await release.wait()

    def handle(request):
        nonlocal calls
        calls += 1
        return httpx.Response(429, headers={'Retry-After': '7'}) if calls == 1 else httpx.Response(200, json=fixture('typesafe'))

    monkeypatch.setattr('ragu.models.typesafe.asyncio.sleep', sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as pool:
        client = TypeSafeClient('key', http_client=pool, retry_times_sec=(2,), rate_max_simultaneous=1)
        task = asyncio.create_task(client.decide('m', state='x', questions=questions))
        await asyncio.wait_for(waiting.wait(), 2)
        # A different call can occupy the same slot while the first call waits.
        await client.decide('m', state='other', questions=questions)
        release.set()
        await task
    assert waits == [7]


def test_retry_after_dates():
    assert _retry_after('bad') == _retry_after('NaN') == _retry_after('-1') == 0
    date = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=30))
    assert 28 <= _retry_after(date) <= 30


async def test_cache_identity_and_isolation(questions):
    calls = 0
    cache = {}

    def handle(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=fixture('typesafe'))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as pool:
        client = TypeSafeClient('key', http_client=pool, cache=cache)
        await client.decide('m', state='x', questions=questions)
        assert (await client.decide('m', state='x', questions=questions)).cache_hit
        await client.decide('other', state='x', questions=questions)
        await client.decide('m', state='other', questions=questions)
        await client.decide('m', state='x', questions=[replace(questions[0], instructions='Different'), *questions[1:]])
        await client.decide('m', state='x', questions=[*questions[:2], replace(questions[2], levels=questions[2].levels[::-1])])
        await client.decide('m', state='x', questions=questions[::-1])
        await client.decide('m', state='x', questions=[questions[0], replace(questions[1], choices=questions[1].choices[::-1]), questions[2]])
        client.url = 'https://proxy.example/prefix'
        await client.decide('m', state='x', questions=questions)
        client.cache_prefix = 'experiment'
        await client.decide('m', state='x', questions=questions)
        assert calls == 9
        assert 'secret' not in repr(cache)
        result = await client.decide('m', state='x', questions=questions)
        result.answers.clear()
        assert len((await client.decide('m', state='x', questions=questions)).answers) == 3


async def test_disk_cache_and_settings(monkeypatch, tmp_path, questions):
    monkeypatch.setattr(Settings, 'cache_path', tmp_path / 'decisions')
    count = 0

    def handle(request):
        nonlocal count
        count += 1
        return httpx.Response(200, json=fixture('typesafe'))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as pool:
        first = TypeSafeClient('key', http_client=pool)
        await first.decide('m', state='x', questions=questions)
        second = TypeSafeClient('other-key', http_client=pool)
        assert (await second.decide('m', state='x', questions=questions)).cache_hit
        explicit = TypeSafeClient('key', http_client=pool, cache={})
        assert not (await explicit.decide('m', state='x', questions=questions)).cache_hit
    assert count == 2


async def test_jev_concurrency_and_cancellation(questions):
    active = maximum = 0
    entered, release = asyncio.Event(), asyncio.Event()

    async def handle(request):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        entered.set()
        try:
            await release.wait()
            return httpx.Response(200, json=fixture('typesafe'))
        finally:
            active -= 1

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as pool:
        client = TypeSafeClient('key', http_client=pool, rate_max_simultaneous=1)
        task = asyncio.create_task(client.decide('m', state='x', questions=questions))
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        await asyncio.gather(*(client.decide('m', state=str(i), questions=questions) for i in range(6)))
        assert maximum == 1 and active == 0


async def test_jev_default_url_and_owned_close():
    client = TypeSafeClient('key')
    assert client.url == 'https://api.typesafe.ai/v1/systemone'
    async with client:
        assert not client.http_client.is_closed
    await client.aclose()
    assert client.http_client.is_closed


async def test_duplicate_jev_answer_and_validation_before_http(questions):
    calls = 0

    def handle(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=b'{"answers":{"predicate":{"type":"noul","noul":0},"predicate":{"type":"noul","noul":1}}}')

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as pool:
        client = TypeSafeClient('key', http_client=pool)
        with pytest.raises(ValueError, match='state'):
            await client.decide('m', state={'x': float('inf')}, questions=questions)
        assert calls == 0
        with pytest.raises(ValueError, match='predicate.*duplicate'):
            await client.decide('m', state='x', questions=questions)
        assert calls == 1


def fake_sdk(raw):
    create = AsyncMock(return_value=SimpleNamespace(model_dump=lambda **kw: raw))
    sdk = SimpleNamespace(base_url='https://example.test/v1/', close=AsyncMock())
    sdk.with_options = Mock(return_value=SimpleNamespace(decisions=SimpleNamespace(create=create)))
    return sdk, create


async def test_openai_wrapper_and_debug_opt_in(questions):
    sdk, create = fake_sdk(fixture('openai'))
    cache = {}
    async with CachedAsyncOpenAI(client=sdk, cache=cache, max_completion_tokens=12) as client:
        result = await client.decide('m', state={'x': None}, questions=questions)
        assert not result.cache_hit
        assert (await client.decide('m', state={'x': None}, questions=questions)).cache_hit
        assert create.await_count == 1
        sdk.with_options.assert_called_with(max_retries=0, timeout=60.0)
        assert set(create.call_args.kwargs) == {'model', 'input', 'questions'}
    sdk.close.assert_not_called()
    debug = {}
    client = CachedAsyncOpenAI(client=sdk, cache={}, debug_errors_storage=debug, retry_times_sec=None)
    create.return_value = SimpleNamespace(model_dump=lambda **kw: {'answers': []})
    with pytest.raises(ValueError):
        await client.decide('m', state='private', questions=questions)
    assert len(debug) == 1 and not client.cache
    assert list(debug.values())[0]['kwargs']['request']['input'] == 'private'


@pytest.mark.parametrize('retry_kind', ['timeout', '429', '401'])
async def test_openai_wrapper_retry_count(retry_kind, questions):
    sdk, create = fake_sdk(fixture('openai'))
    req = httpx.Request('POST', 'https://example.test/v1/decisions')
    if retry_kind == 'timeout':
        error = openai.APITimeoutError(request=req)
    else:
        status = int(retry_kind)
        cls = openai.RateLimitError if status == 429 else openai.AuthenticationError
        error = cls('error', response=httpx.Response(status, request=req), body={})
    create.side_effect = [error, SimpleNamespace(model_dump=lambda **kw: fixture('openai'))]
    client = CachedAsyncOpenAI(client=sdk, retry_times_sec=(0, 0, 0))
    if retry_kind == '401':
        with pytest.raises(openai.AuthenticationError):
            await client.decide('m', state='x', questions=questions)
        assert create.await_count == 1
    else:
        await client.decide('m', state='x', questions=questions)
        assert create.await_count == 2


async def test_openai_shared_limits_and_cancellation(questions):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    active = maximum = 0
    release, entered = asyncio.Event(), asyncio.Event()

    async def request(kind):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        entered.set()
        try:
            await release.wait()
            if kind == 'decision':
                return SimpleNamespace(model_dump=lambda **kw: fixture('openai'))
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='ok'))])
        finally:
            active -= 1

    sdk, create = fake_sdk(fixture('openai'))

    async def decide(**kwargs):
        return await request('decision')

    async def chat(**kwargs):
        return await request('chat')

    create.side_effect = decide
    sdk.chat = SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(side_effect=chat)))
    client = CachedAsyncOpenAI(client=sdk, rate_max_simultaneous=1, cache={})
    task = asyncio.create_task(client.decide('m', state='x', questions=questions))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    await asyncio.gather(
        client.decide('m', state='x', questions=questions),
        client.chat_completion('m', conversation=[{'role': 'user', 'content': 'x'}]),
    )
    assert maximum == 1 and active == 0
    assert len(client.cache) == 2  # Chat and decision entries never collide.


async def test_openai_debug_disk(tmp_path, questions):
    sdk, create = fake_sdk({'answers': []})
    client = CachedAsyncOpenAI(client=sdk, debug_errors_storage=tmp_path / 'debug', retry_times_sec=None)
    initial_keys = set(client.debug_errors_storage)
    with pytest.raises(ValueError):
        await client.decide('m', state='private', questions=questions)
    new_keys = set(client.debug_errors_storage) - initial_keys
    assert len(new_keys) == 1
    saved = client.debug_errors_storage[new_keys.pop()]
    assert saved['kwargs']['request']['input'] == 'private'


@pytest.mark.parametrize('url', ['relative/path', 'ftp://example/op', 'https://key@example/op', 'https://example/op#fragment'])
def test_invalid_operation_url(url):
    with pytest.raises(ValueError, match='url'):
        TypeSafeClient('key', url=url)


@pytest.mark.parametrize('schedule,total', [(None, 1), ((), 1), ((0, 0, 0), 4)])
async def test_openai_exhaustion_and_disabled_retries(schedule, total, questions):
    sdk, create = fake_sdk(fixture('openai'))
    request = httpx.Request('POST', 'https://example.test/v1/decisions')
    create.side_effect = openai.RateLimitError('error', response=httpx.Response(429, request=request), body={})
    client = CachedAsyncOpenAI(client=sdk, retry_times_sec=schedule, cache={})
    with pytest.raises(openai.RateLimitError):
        await client.decide('m', state='x', questions=questions)
    assert create.await_count == total
    assert not client.cache


async def test_decision_cache_excludes_credentials_and_other_operations(questions):
    sdk, create = fake_sdk(fixture('openai'))
    client = CachedAsyncOpenAI(client=sdk, cache={})
    client._uncached_chat_completion = AsyncMock(return_value='ok')
    client._uncached_embed_text = AsyncMock(return_value=[0.1])
    client._uncached_score = AsyncMock(return_value=[(0, 0.2)])
    await client.decide('m', state='x', questions=questions)
    await client.chat_completion('m', conversation=[{'role': 'user', 'content': 'x'}])
    await client.embed_text('m', text='x')
    await client.score('m', text_1='x', text_2=['y'])
    assert len(client.cache) == 4
    assert (await client.decide('m', state='x', questions=questions)).cache_hit
    assert create.await_count == 1


@pytest.mark.parametrize('field,value', [
    ('api_key', ' '), ('api_key', 123),
    ('timeout', True), ('timeout', '60'), ('timeout', float('inf')),
    ('rate_min_delay', -1), ('rate_min_delay', float('nan')),
    ('rate_max_per_minute', 1.5), ('rate_max_simultaneous', True),
    ('retry_times_sec', [False]), ('retry_times_sec', [-1]), ('retry_times_sec', [float('inf')]),
])
def test_client_config_is_strict(field, value):
    params = {'api_key': 'key', field: value}
    with pytest.raises(ValueError, match=field):
        TypeSafeClient(**params)


@pytest.mark.parametrize('options,expected', [
    ({}, 60.0), ({'timeout': 3.0}, 3.0), ({'embed_timeout': 4.0}, 4.0),
    ({'timeout': 3.0, 'embed_timeout': 4.0}, 3.0),
    ({'timeout': None}, None), ({'embed_timeout': None}, None),
    ({'timeout': None, 'embed_timeout': 4.0}, None),
])
async def test_decisions_share_timeout(options, expected, questions):
    sdk, create = fake_sdk(fixture('openai'))
    sdk.timeout = 12.0
    async with CachedAsyncOpenAI(client=sdk, retry_times_sec=None, **options) as client:
        await client.decide('m', state='x', questions=questions)
        expected_options = {'max_retries': 0}
        if expected is not None:
            expected_options['timeout'] = expected
        sdk.with_options.assert_called_once_with(**expected_options)
        assert sdk.timeout == 12.0
    create.assert_awaited_once()
