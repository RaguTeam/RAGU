"""Opt-in functional check against OpenRouter using credentials from .env."""

import json
import os
from dataclasses import asdict
from pathlib import Path

import httpx
import pytest
from dotenv import dotenv_values

from ragu import (
    ChoiceAnswer, ChoiceOption, ChoiceQuestion, PredicateAnswer, PredicateQuestion,
    ScoreAnswer, ScoreLevel, ScoreQuestion, TypeSafeClient,
)


@pytest.mark.integration
async def test_openrouter_mixed_decisions_and_cache():
    if os.environ.get('RAGU_DECISIONS_SMOKE') != '1':
        pytest.skip('Для реального вызова задайте RAGU_DECISIONS_SMOKE=1')
    config = dotenv_values(Path(__file__).resolve().parents[2] / '.env')
    key = config.get('OPENROUTER_API_KEY') or os.environ.get('OPENROUTER_API_KEY')
    if not key:
        pytest.skip('В .env или окружении отсутствует OPENROUTER_API_KEY')
    model = (config.get('OPENROUTER_DECISION_MODEL')
             or os.environ.get('OPENROUTER_DECISION_MODEL') or 'typesafe/jev-1.14')
    state = {
        'claim_a': 'На 1 января 2025 года у компании был ровно один филиал.',
        'claim_b': 'На 1 января 2025 года у той же компании было ровно три филиала.',
        'context': {'same_company': True, 'date': '2025-01-01', 'notes': [None, 'Филиалы считаются одинаково.']},
    }
    questions = [
        PredicateQuestion('contradiction', 'Противоречат ли утверждения друг другу? '
                          'Противоречие означает, что оба утверждения не могут быть истинны одновременно.'),
        ChoiceQuestion('relation', 'Выбери отношение между утверждениями.', [
            ChoiceOption('contradictory', 'Утверждения противоречат друг другу.'),
            ChoiceOption('compatible', 'Оба утверждения могут быть истинны одновременно.'),
        ]),
        ScoreQuestion('contradiction_strength', 'Оцени силу противоречия при одинаковых компании и дате.', [
            ScoreLevel('none', 'Противоречия нет.'),
            ScoreLevel('uncertain', 'Недостаточно информации.'),
            ScoreLevel('clear', 'Утверждения явно несовместимы.'),
        ]),
    ]
    requests = []

    async def track_request(request: httpx.Request) -> None:
        requests.append(str(request.url))

    async with httpx.AsyncClient(event_hooks={'request': [track_request]}) as pool:
        async with TypeSafeClient(
            key, url='https://openrouter.ai/api/alpha/decisions', http_client=pool,
            timeout=15.0, retry_times_sec=None, cache={},
        ) as client:
            result = await client.decide(model, state=state, questions=questions)
            cached = await client.decide(model, state=state, questions=questions)
        assert not pool.is_closed

    assert requests == ['https://openrouter.ai/api/alpha/decisions']
    assert list(result.answers) == ['contradiction', 'relation', 'contradiction_strength']
    predicate = result.answers['contradiction']
    choice = result.answers['relation']
    score = result.answers['contradiction_strength']
    assert isinstance(predicate, PredicateAnswer) and 0 <= predicate.probability <= 1
    assert isinstance(choice, ChoiceAnswer) and choice.choice in {'contradictory', 'compatible'}
    assert isinstance(score, ScoreAnswer) and 0 <= score.score <= 2
    assert set(choice.probabilities) == {'contradictory', 'compatible'}
    assert set(score.probabilities) == {0, 1, 2}
    for answer in (choice, score):
        assert 0 <= answer.confidence <= 1
        assert all(0 <= probability <= 1 for probability in answer.probabilities.values())
        assert abs(sum(answer.probabilities.values()) - 1) <= 0.01
    assert not result.cache_hit and cached.cache_hit
    assert cached.answers == result.answers and cached.usage == result.usage
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
