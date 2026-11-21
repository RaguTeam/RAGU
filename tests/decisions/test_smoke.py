"""Explicitly enabled live checks, excluded from the normal offline suite."""

import os

import pytest

from ragu import CachedAsyncOpenAI, ChoiceAnswer, PredicateAnswer, ScoreAnswer, TypeSafeClient


@pytest.mark.integration
@pytest.mark.parametrize('route,key_env,model_env,url', [
    ('typesafe', 'TYPESAFE_API_KEY', 'TYPESAFE_DECISION_MODEL', 'https://api.typesafe.ai/v1/systemone'),
    ('openrouter', 'OPENROUTER_API_KEY', 'OPENROUTER_DECISION_MODEL', 'https://openrouter.ai/api/alpha/decisions'),
    ('openai', 'OPENAI_API_KEY', 'OPENAI_DECISION_MODEL', None),
])
async def test_live_decisions(route, key_env, model_env, url, questions):
    if os.environ.get('RAGU_DECISIONS_SMOKE') != '1':
        pytest.skip('Set RAGU_DECISIONS_SMOKE=1 to enable billable live calls')
    key, model = os.environ.get(key_env), os.environ.get(model_env)
    if not key or not model:
        pytest.skip(f'Requires {key_env} and {model_env}')
    client = CachedAsyncOpenAI(api_key=key, cache={}) if route == 'openai' else TypeSafeClient(key, url=url, cache={})
    async with client:
        result = await client.decide(model, state='The payment failed twice and blocks all purchases.', questions=questions)
    assert isinstance(result.answers['predicate'], PredicateAnswer)
    assert isinstance(result.answers['choice'], ChoiceAnswer)
    assert isinstance(result.answers['score'], ScoreAnswer)
    assert 0 <= result.answers['predicate'].probability <= 1
    assert 0 <= result.answers['score'].score <= 2
