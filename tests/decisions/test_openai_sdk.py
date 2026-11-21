"""Offline acceptance checks against the actual supported SDK and HTTP transport."""

import asyncio
import json
from pathlib import Path

import openai
import pytest

from ragu import CachedAsyncOpenAI, PredicateQuestion

# The suite must exercise the real Decisions resource; older SDKs cannot do so.
pytestmark = pytest.mark.skipif(
    tuple(int(x) for x in openai.__version__.split('.')[:2]) < (3, 57),
    reason='Requires installed openai>=3.57.0 with Decisions resource',
)


@pytest.fixture
def sdk_http():
    return pytest.importorskip('httpx2')


def payload():
    return json.loads((Path(__file__).parent / 'fixtures' / 'openai.json').read_text())


@pytest.mark.parametrize('scenario,expected_calls', [
    ('ok', 1), ('429', 2), ('timeout', 2), ('exhausted', 4), ('401', 1), ('400', 1), ('json', 1), ('invalid', 1),
])
async def test_sdk_wire_retry_settings_and_ownership(sdk_http, questions, scenario, expected_calls):
    requests = []

    def handle(request):
        requests.append(request)
        assert str(request.url) == 'https://proxy.example/prefix/v1/decisions'
        assert request.headers['authorization'] == 'Bearer external-key'
        assert request.headers['openai-project'] == 'project-test'
        assert request.headers['openai-organization'] == 'org-test'
        assert request.headers['x-custom'] == 'inherited'
        assert request.extensions['timeout']['read'] == 3.0
        body = json.loads(request.content)
        assert set(body) == {'model', 'input', 'questions'}
        assert body['input'][0]['content'][0] == {'type': 'input_json', 'data': {'текст': [1, None]}}
        assert [q['name'] for q in body['questions']] == ['predicate', 'choice', 'score']
        if scenario == 'timeout' and len(requests) == 1:
            raise sdk_http.ReadTimeout('test', request=request)
        if scenario in ('401', '400', 'exhausted') or (scenario == '429' and len(requests) == 1):
            status = 429 if scenario in ('429', 'exhausted') else int(scenario)
            return sdk_http.Response(status, json={'error': {'message': 'test', 'type': 'test'}})
        if scenario == 'json':
            return sdk_http.Response(200, content=b'{broken', headers={'content-type': 'application/json'})
        if scenario == 'invalid':
            return sdk_http.Response(200, json={'answers': []})
        return sdk_http.Response(200, json=payload())

    async with sdk_http.AsyncClient(transport=sdk_http.MockTransport(handle)) as pool:
        async with openai.AsyncOpenAI(
            api_key='external-key', base_url='https://proxy.example/prefix/v1/',
            http_client=pool, project='project-test', organization='org-test',
            default_headers={'X-Custom': 'inherited'}, max_retries=6,
        ) as sdk:
            async with CachedAsyncOpenAI(client=sdk, retry_times_sec=(0, 0, 0), timeout=3, cache={}) as client:
                if scenario in ('ok', '429', 'timeout'):
                    result = await client.decide('m', state={'текст': [1, None]}, questions=questions)
                    assert list(result.answers) == ['predicate', 'choice', 'score']
                    assert (await client.decide('m', state={'текст': [1, None]}, questions=questions)).cache_hit
                else:
                    with pytest.raises((openai.APIStatusError, ValueError)) as error:
                        await client.decide('m', state={'текст': [1, None]}, questions=questions)
                    assert not client.cache
                    if scenario in ('401', '400'):
                        assert error.value.status_code == int(scenario)
                assert len(requests) == expected_calls
                assert sdk.max_retries == 6
            assert not sdk.is_closed() and not pool.is_closed


async def test_chat_and_decisions_share_quota(sdk_http):
    active = maximum = 0

    async def handle(request):
        nonlocal active, maximum
        active += 1
        maximum = max(active, maximum)
        try:
            await asyncio.sleep(0.01)
            if request.url.path.endswith('/decisions'):
                return sdk_http.Response(200, json={
                    'model': 'm', 'answers': [{'name': 'p', 'type': 'predicate', 'probability': 0}],
                })
            return sdk_http.Response(200, json={
                'id': 'chat', 'model': 'm', 'created': 0, 'object': 'chat.completion',
                'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'ok'}}],
            })
        finally:
            active -= 1

    async with sdk_http.AsyncClient(transport=sdk_http.MockTransport(handle)) as pool:
        async with openai.AsyncOpenAI(api_key='key', http_client=pool, max_retries=0) as sdk:
            async with CachedAsyncOpenAI(client=sdk, rate_max_simultaneous=1) as client:
                await asyncio.gather(
                    client.chat_completion('m', conversation=[{'role': 'user', 'content': 'hello'}]),
                    client.decide('m', state='x', questions=[PredicateQuestion('p', 'True?')]),
                )
    assert maximum == 1


async def test_owned_sdk_close():
    client = CachedAsyncOpenAI(api_key='key')
    async with client:
        assert not client.client.is_closed()
    await client.aclose()
    assert client.client.is_closed()
