from copy import deepcopy
from dataclasses import replace

import pytest

from ragu import (
    ChoiceAnswer, ChoiceOption, PredicateAnswer, PredicateQuestion,
    RefusalAnswer, ScoreAnswer, ScoreLevel,
)
from ragu.models.decisions.jev import _parse_jev
from ragu.models.decisions.openai import _parse_openai


def codec(case):
    return _parse_openai if case == 'openai' else _parse_jev


def named(raw, name):
    answers = raw['answers']
    return next(a for a in answers if a['name'] == name) if isinstance(answers, list) else answers[name]


@pytest.mark.parametrize('state', ['Состояние', {'данные': [1, None, {'x': False, 'n': 1.5}]}, [None, 'текст']])
async def test_mixed_conversion(provider_case, decision_client, questions, state):
    provider, raw = provider_case
    await decision_client.decide('model', state=state, questions=questions)
    body = decision_client._cached_decide.call_args.kwargs['request']
    if provider == 'openai':
        assert body['input'] == (state if isinstance(state, str) else [
            {'role': 'user', 'content': [{'type': 'input_json', 'data': state}]},
        ])
        assert [q['type'] for q in body['questions']] == ['predicate', 'choice', 'score']
        assert body['questions'][1]['choices'][0] == {'value': 'a', 'description': 'First'}
    else:
        assert body['state'] == state
        assert list(body['questions']) == ['predicate', 'choice', 'score']
        assert body['questions']['choice']['criteria'] == {'a': 'First', 'b': 'Second'}
        assert body['questions']['score']['criteria'] == ['low — Can wait', 'medium — This week', 'high — Now']
    result = codec(provider)(raw, 'model', questions)
    assert list(result.answers) == [q.name for q in questions]
    assert isinstance(result.answers['predicate'], PredicateAnswer)
    assert result.answers['predicate'].probability == (0.96 if provider == 'openrouter' else 0.0)
    assert result.answers['choice'] == ChoiceAnswer('b', {'a': 0.16, 'b': 0.84}, 0.75)
    assert result.answers['score'] == ScoreAnswer(1.99, {0: 0, 1: 0.01, 2: 0.99}, 0.99)
    assert result.requested_model == 'model'
    assert result.resolved_model == raw['model']
    assert result.usage == raw['usage']
    assert result.provider == raw.get('provider')
    assert result.id == raw.get('id')


def test_partial_refusal(provider_case, questions):
    provider, raw = provider_case
    answer = named(raw, 'choice')
    answer.clear()
    answer.update(type='refusal', name='choice', reason='declined')
    result = codec(provider)(raw, 'm', questions)
    assert isinstance(result.answers['choice'], RefusalAnswer)
    assert result.answers['choice'].details == {'reason': 'declined'}
    assert isinstance(result.answers['score'], ScoreAnswer)


@pytest.mark.parametrize('mutation,field', [
    (lambda a: a.update(type='score'), 'type'),
    (lambda a: a.update(confidence=2), 'confidence'),
    (lambda a: a.update(choice='unknown'), 'choice'),
    (lambda a: a.update(probabilities={}), 'probabilities'),
])
def test_invalid_choice_response(provider_case, questions, mutation, field):
    provider, raw = provider_case
    mutation(named(raw, 'choice'))
    with pytest.raises(ValueError, match=field) as error:
        codec(provider)(raw, 'm', questions)
    assert 'choice' in str(error.value)
    assert ('OpenAI' if provider == 'openai' else 'Jev') in str(error.value)


@pytest.mark.parametrize('number', [True, '0.5', None, -0.1, 1.1, float('nan'), float('inf')])
def test_invalid_predicate_numbers(provider_case, questions, number):
    provider, raw = provider_case
    field = 'probability' if provider == 'openai' else 'noul'
    named(raw, 'predicate')[field] = number
    with pytest.raises(ValueError, match=field):
        codec(provider)(raw, 'm', questions)


@pytest.mark.parametrize('mode', ['missing', 'extra'])
def test_response_names(provider_case, questions, mode):
    provider, raw = provider_case
    if isinstance(raw['answers'], list):
        if mode == 'missing':
            raw['answers'].pop()
        else:
            raw['answers'].append({'name': 'unknown', 'type': 'refusal'})
    elif mode == 'missing':
        raw['answers'].pop('predicate')
    else:
        raw['answers']['unknown'] = {'type': 'refusal'}
    with pytest.raises(ValueError, match='name'):
        codec(provider)(raw, 'm', questions)


@pytest.mark.parametrize('state', [None, 3, True, {1: 'bad'}, {'x': float('nan')}, [float('inf')], {'x': (1, 2)}, {'x': object()}])
async def test_invalid_state(decision_client, questions, state):
    with pytest.raises(ValueError, match='state') as error:
        await decision_client.decide('m', state=state, questions=questions)
    assert 'input_value=' not in str(error.value)
    decision_client._cached_decide.assert_not_awaited()


async def test_cyclic_state_rejected(decision_client, questions):
    state = []
    state.append(state)
    with pytest.raises(ValueError, match='state'):
        await decision_client.decide('m', state=state, questions=questions)


@pytest.mark.parametrize('field', ['name', 'instructions', 'duplicate_name', 'choices', 'description', 'levels', 'label'])
async def test_invalid_question_fields(decision_client, questions, field):
    if field in ('name', 'instructions'):
        questions[0] = replace(questions[0], **{field: ''})
    elif field == 'duplicate_name':
        questions[1] = replace(questions[1], name=questions[0].name)
    elif field == 'choices':
        questions[1] = replace(questions[1], choices=[ChoiceOption('a', 'A'), ChoiceOption('a', 'B')])
    elif field == 'description':
        questions[1] = replace(questions[1], choices=[ChoiceOption('a', '')])
    elif field == 'levels':
        questions[2] = replace(questions[2], levels=[ScoreLevel('a', 'A')])
    else:
        questions[2] = replace(questions[2], levels=[ScoreLevel('', 'A'), ScoreLevel('b', 'B')])
    with pytest.raises(ValueError, match='questions'):
        await decision_client.decide('m', state='', questions=questions)


async def test_empty_questions(decision_client):
    with pytest.raises(ValueError, match='questions'):
        await decision_client.decide('m', state='', questions=[])


def test_rounding_tolerance(provider_case, questions):
    provider, raw = provider_case
    answer = named(raw, 'score')
    if provider == 'openai':
        for v in answer['probabilities']:
            v['probability'] = 0.333
    else:
        answer['probabilities'] = {str(i): 0.333 for i in range(3)}
    codec(provider)(raw, 'm', questions)
    if provider == 'openai':
        answer['probabilities'][0]['probability'] = 0.5
    else:
        answer['probabilities']['0'] = 0.5
    with pytest.raises(ValueError, match='probabilities'):
        codec(provider)(raw, 'm', questions)


def test_optional_metadata(questions):
    raw = {'answers': {q.name: {'type': 'refusal'} for q in questions}}
    result = _parse_jev(raw, 'm', questions)
    assert result.usage is result.id is result.provider is result.resolved_model is None


@pytest.mark.parametrize('change,field', [
    (lambda qs: [], 'questions'),
    (lambda qs: [qs[0], qs[0]], 'name'),
    (lambda qs: [replace(qs[0], name=' ')], 'name'),
    (lambda qs: [replace(qs[0], instructions=None)], 'instructions'),
    (lambda qs: [replace(qs[1], choices=[])], 'choices'),
    (lambda qs: [replace(qs[1], choices=[ChoiceOption('a', 'x'), ChoiceOption('a', 'y')])], 'value'),
    (lambda qs: [replace(qs[1], choices=[ChoiceOption('a', ''), ChoiceOption('b', 'x')])], 'description'),
    (lambda qs: [replace(qs[2], levels=[ScoreLevel('a', 'x')])], 'levels'),
    (lambda qs: [replace(qs[2], levels=[ScoreLevel('', 'x'), ScoreLevel('b', 'y')])], 'label'),
])
async def test_invalid_questions(decision_client, questions, change, field):
    with pytest.raises(ValueError, match=field):
        await decision_client.decide('m', state='state', questions=change(questions))
    decision_client._cached_decide.assert_not_awaited()


@pytest.mark.parametrize('provider_case', ['openai'], indirect=True)
async def test_known_openai_limits(decision_client):
    with pytest.raises(ValueError, match='questions'):
        await decision_client.decide('m', state='state', questions=[PredicateQuestion(str(i), 'True?') for i in range(401)])
    from ragu import ChoiceQuestion
    with pytest.raises(ValueError, match='choices'):
        await decision_client.decide('m', state='state', questions=[ChoiceQuestion('c', 'Choose', [ChoiceOption('x', 'x')])])


async def test_cyclic_state(decision_client, questions):
    state = []
    state.append(state)
    with pytest.raises(ValueError, match='(?s)state.*cyclic'):
        await decision_client.decide('m', state=state, questions=questions)


@pytest.mark.parametrize('field,value', [('confidence', float('nan')), ('score', True), ('score', -1), ('score', 3)])
def test_invalid_score(provider_case, questions, field, value):
    provider, raw = provider_case
    named(raw, 'score')[field] = value
    with pytest.raises(ValueError, match=field):
        codec(provider)(raw, 'm', questions)


def test_duplicate_distribution_openai(questions):
    import json
    from pathlib import Path
    raw = json.loads((Path(__file__).parent / 'fixtures' / 'openai.json').read_text())
    named(raw, 'choice')['probabilities'].append({'value': 'a', 'probability': 0})
    with pytest.raises(ValueError, match='(?s)probabilities.*duplicate'):
        _parse_openai(raw, 'm', questions)


def test_duplicate_openai_answer(questions):
    import json
    from pathlib import Path
    raw = json.loads((Path(__file__).parent / 'fixtures' / 'openai.json').read_text())
    raw['answers'].append(deepcopy(raw['answers'][0]))
    with pytest.raises(ValueError, match='name.*duplicate'):
        _parse_openai(raw, 'm', questions)


@pytest.mark.parametrize('field,value', [
    ('confidence', True), ('confidence', '0.75'), ('choice', 1),
])
def test_response_fields_are_strict(provider_case, questions, field, value):
    provider, raw = provider_case
    named(raw, 'choice')[field] = value
    with pytest.raises(ValueError, match=field):
        codec(provider)(raw, 'm', questions)


@pytest.mark.parametrize('value', [True, '0.5', float('inf')])
def test_distribution_values_are_strict(provider_case, questions, value):
    provider, raw = provider_case
    probabilities = named(raw, 'choice')['probabilities']
    if provider == 'openai':
        probabilities[0]['probability'] = value
    else:
        probabilities['a'] = value
    with pytest.raises(ValueError, match='probabilities'):
        codec(provider)(raw, 'm', questions)


@pytest.mark.parametrize('field,value', [
    ('model', 1), ('provider', False), ('id', 2),
    ('usage', []), ('usage', {'cost': float('nan')}), ('usage', {'details': (1, 2)}),
])
def test_metadata_is_validated(provider_case, questions, field, value):
    provider, raw = provider_case
    raw[field] = value
    with pytest.raises(ValueError, match=field):
        codec(provider)(raw, 'm', questions)


@pytest.mark.parametrize('provider_case', ['openai'], indirect=True)
@pytest.mark.parametrize('value', [True, '0', 0.0])
def test_openai_score_indices_are_strict(provider_case, questions, value):
    provider, raw = provider_case
    named(raw, 'score')['probabilities'][0]['value'] = value
    with pytest.raises(ValueError, match='probabilities.*value'):
        _parse_openai(raw, 'm', questions)


@pytest.mark.parametrize('provider_case', ['typesafe', 'openrouter'], indirect=True)
@pytest.mark.parametrize('value', ['00', '-1', '0.0', 'x'])
def test_jev_score_indices_are_canonical(provider_case, questions, value):
    provider, raw = provider_case
    probabilities = named(raw, 'score')['probabilities']
    probabilities[value] = probabilities.pop('0')
    with pytest.raises(ValueError, match='probabilities'):
        _parse_jev(raw, 'm', questions)


@pytest.mark.parametrize('provider_case', ['openai'], indirect=True)
def test_openai_errors_identify_arbitrary_question_names(provider_case, questions):
    provider, raw = provider_case
    question = replace(questions[0], name='contradiction')
    answer = named(raw, 'predicate')
    answer.update(name=question.name, probability='not-a-number')
    with pytest.raises(ValueError, match='(?s)OpenAI.*contradiction.*probability') as error:
        _parse_openai(raw, 'm', [question, *questions[1:]])
    assert 'not-a-number' not in str(error.value)


@pytest.mark.parametrize('field,value', [
    ('choices', [{'value': 'a', 'description': 'First'}]),
    ('choices', [ChoiceOption(1, 'First')]),
    ('levels', [{'label': 'low', 'description': 'Low'}, ScoreLevel('high', 'High')]),
])
async def test_question_attributes_are_validated(decision_client, questions, field, value):
    question = questions[1] if field == 'choices' else questions[2]
    with pytest.raises(ValueError, match=field):
        await decision_client.decide('m', state='state', questions=[replace(question, **{field: value})])


def test_well_formed_answer_with_wrong_kind(provider_case, questions):
    provider, raw = provider_case
    replacement = deepcopy(named(raw, 'score'))
    replacement['name'] = 'choice'
    answer = named(raw, 'choice')
    answer.clear()
    answer.update(replacement)
    with pytest.raises(ValueError, match='choice.*type'):
        codec(provider)(raw, 'm', questions)


@pytest.mark.parametrize('provider_case', ['openai'], indirect=True)
@pytest.mark.parametrize('answers', [
    {'predicate': {'type': 'predicate', 'probability': 0}},
    [{'name': 123, 'type': 'refusal'}],
    [{'type': 'refusal'}],
    [None],
])
def test_malformed_openai_answer_envelope(provider_case, questions, answers):
    _, raw = provider_case
    raw['answers'] = answers
    with pytest.raises(ValueError, match='(?s)OpenAI.*answers'):
        _parse_openai(raw, 'm', questions)
