"""Verify shared timeouts at the HTTP boundary without network access."""

import json
from types import SimpleNamespace

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic import BaseModel

from ragu import CachedAsyncOpenAI


class _Output(BaseModel):
    value: str


@pytest.mark.parametrize('options,sdk_timeout,expected', [
    ({}, 12.0, 60.0),
    ({'timeout': 3.0}, 12.0, 3.0),
    ({'embed_timeout': 4.0}, 12.0, 4.0),
    ({'timeout': 3.0, 'embed_timeout': 4.0}, 12.0, 3.0),
    ({'timeout': None}, 12.0, 12.0),
    ({'embed_timeout': None}, 12.0, 12.0),
    ({'timeout': None, 'embed_timeout': 4.0}, 12.0, 12.0),
    ({'timeout': None}, None, None),
])
async def test_shared_sdk_timeout(options, sdk_timeout, expected):
    requests = []

    def handle(request):
        requests.append(request)
        assert request.extensions['timeout']['read'] == expected
        body = json.loads(request.content)
        if request.url.path.endswith('/embeddings'):
            inputs = body['input'] if isinstance(body['input'], list) else [body['input']]
            return httpx.Response(200, json={
                'object': 'list', 'model': 'm', 'data': [
                    {'object': 'embedding', 'index': i, 'embedding': [0.1, 0.2]}
                    for i in range(len(inputs))
                ],
            })
        if body.get('stream'):
            event = {
                'id': 'stream', 'object': 'chat.completion.chunk', 'created': 0, 'model': 'm',
                'choices': [{'index': 0, 'finish_reason': None, 'delta': {'content': 'ok'}}],
            }
            return httpx.Response(200, text=f'data: {json.dumps(event)}\n\ndata: [DONE]\n\n',
                                  headers={'content-type': 'text/event-stream'})
        message = {'role': 'assistant', 'content': '{"value":"ok"}'}
        if 'tools' in body:
            message['tool_calls'] = [{
                'id': 'call', 'type': 'function',
                'function': {'name': '_Output', 'arguments': '{"value":"ok"}'},
            }]
        return httpx.Response(200, json={
            'id': 'chat', 'object': 'chat.completion', 'created': 0, 'model': 'm',
            'choices': [{'index': 0, 'finish_reason': 'stop', 'message': message}],
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as pool:
        async with AsyncOpenAI(api_key='key', http_client=pool, timeout=sdk_timeout, max_retries=0) as sdk:
            # Keep SDK platform discovery out of the worker thread in this HTTP test.
            sdk._platform = 'Linux'
            async with CachedAsyncOpenAI(client=sdk, retry_times_sec=None, **options) as client:
                messages = [{'role': 'user', 'content': 'hello'}]
                assert await client.chat_completion('m', messages) == '{"value":"ok"}'
                assert await client.chat_completion('m', messages, output_schema=_Output) == _Output(value='ok')
                assert await client.chat_completion('m', messages, output_schema=_Output, as_tool=True) == _Output(value='ok')
                assert [part async for part in client.stream_chat_completion('m', messages)] == ['ok']
                assert await client.embed_text('m', 'text') == [0.1, 0.2]
                assert await client.embed_texts('m', ['one', 'two']) == [[0.1, 0.2], [0.1, 0.2]]
                assert sdk.timeout == sdk_timeout
    assert len(requests) == 6


@pytest.mark.parametrize('options,expected', [
    ({}, 60.0), ({'timeout': 3.0}, 3.0), ({'embed_timeout': 4.0}, 4.0),
    ({'timeout': 3.0, 'embed_timeout': 4.0}, 3.0), ({'timeout': None}, 12.0),
    ({'embed_timeout': None}, 12.0),
])
async def test_shared_scoring_timeout(monkeypatch, options, expected):
    original_client = httpx.AsyncClient
    requests = []

    def handle(request):
        requests.append(request)
        assert request.extensions['timeout']['read'] == expected
        return httpx.Response(200, json={'data': [{'index': 0, 'score': 0.9}]})

    def pool_factory(**kwargs):
        return original_client(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr('ragu.models.openai.httpx.AsyncClient', pool_factory)
    sdk = SimpleNamespace(api_key='key', base_url='https://example.test/v1/', timeout=httpx.Timeout(12.0))
    async with CachedAsyncOpenAI(client=sdk, retry_times_sec=None, **options) as client:
        assert await client.score('m', 'query', ['document']) == [(0, 0.9)]
        assert sdk.timeout.read == 12.0
    assert len(requests) == 1


def test_legacy_timeout_position_and_attribute():
    client = CachedAsyncOpenAI(None, None, SimpleNamespace(), None, None, None,
                               None, None, None, 'openai', None, 7.0)
    assert client.timeout == client.embed_timeout == 7.0
    client.embed_timeout = 8.0
    assert client.timeout == 8.0
    client.timeout = None
    assert client.embed_timeout is None
