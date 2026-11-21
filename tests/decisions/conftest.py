import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ragu import (
    CachedAsyncOpenAI, ChoiceOption, ChoiceQuestion, DecisionResult,
    PredicateQuestion, ScoreLevel, ScoreQuestion, TypeSafeClient,
)


@pytest.fixture
def questions():
    return [
        PredicateQuestion('predicate', 'Is this true?'),
        ChoiceQuestion('choice', 'Choose one', [ChoiceOption('a', 'First'), ChoiceOption('b', 'Second')]),
        ScoreQuestion('score', 'Rate urgency', [
            ScoreLevel('low', 'Can wait'), ScoreLevel('medium', 'This week'), ScoreLevel('high', 'Now'),
        ]),
    ]


@pytest.fixture(params=['typesafe', 'openrouter', 'openai'])
def provider_case(request):
    name = request.param
    fixture = json.loads((Path(__file__).parent / 'fixtures' / f'{name}.json').read_text())
    return name, fixture


@pytest.fixture
async def decision_client(provider_case, monkeypatch):
    provider, _ = provider_case
    if provider == 'openai':
        client = CachedAsyncOpenAI(api_key='test-key')
    else:
        url = ('https://openrouter.ai/api/alpha/decisions' if provider == 'openrouter'
               else 'https://api.typesafe.ai/v1/systemone')
        client = TypeSafeClient('test-key', url=url)
    monkeypatch.setattr(client, '_cached_decide', AsyncMock(
        return_value=DecisionResult(answers={}, requested_model='model'),
    ))
    async with client:
        yield client
