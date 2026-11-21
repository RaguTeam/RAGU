"""Strict Pydantic schemas for decision arguments and provider responses."""

import math
from collections.abc import Sequence
from typing import Annotated, Any, Generic, Literal, TypeVar

from pydantic import (
    AfterValidator,
    AnyHttpUrl,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Discriminator,
    Field,
    InstanceOf,
    JsonValue,
    StringConstraints,
    Tag,
    TypeAdapter,
    ValidationInfo,
    field_validator,
)

from ragu.models.decisions.types import (
    ChoiceOption,
    ChoiceQuestion,
    PredicateQuestion,
    ScoreLevel,
    ScoreQuestion,
)

# Allow the last decimal place in provider distributions rounded to three digits.
PROBABILITY_SUM_TOLERANCE = 0.01
_Text = Annotated[str, StringConstraints(min_length=1, pattern=r'\S')]
_Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
_PositiveFloat = Annotated[float, Field(gt=0)]
_PositiveInt = Annotated[int, Field(gt=0)]
_ValueT = TypeVar('_ValueT', str, int)


class _Model(BaseModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False, hide_input_in_errors=True)


class _AttributesModel(_Model):
    model_config = ConfigDict(from_attributes=True)


class _TypeSafeConfig(_Model):
    api_key: _Text
    url: AnyHttpUrl
    timeout: _PositiveFloat | None
    rate_min_delay: _PositiveFloat | None
    rate_max_per_minute: _PositiveInt | None
    rate_max_simultaneous: _PositiveInt | None
    retry_times_sec: Sequence[Annotated[float, Field(ge=0)]] | None

    @field_validator('url')
    @classmethod
    def _url(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        if value.username or value.password or value.fragment:
            raise ValueError('expected operation URL without credentials or fragment')
        return value


class _Option(_AttributesModel):
    value: _Text
    description: _Text


class _Level(_AttributesModel):
    label: _Text
    description: _Text


class _Question(_AttributesModel):
    name: _Text
    instructions: _Text


class _ChoiceQuestion(_Question):
    choices: Annotated[Sequence[Annotated[
        _Option, BeforeValidator(TypeAdapter(InstanceOf[ChoiceOption]).validate_python),
    ]], Field(min_length=1)]

    @field_validator('choices')
    @classmethod
    def _choices(cls, value: Sequence[_Option], info: ValidationInfo) -> Sequence[_Option]:
        if info.context['protocol'] == 'openai' and not 2 <= len(value) <= 255:
            raise ValueError('OpenAI accepts 2–255 choices')
        if len({option.value for option in value}) != len(value):
            raise ValueError('value: duplicate choice value')
        return value


class _ScoreQuestion(_Question):
    levels: Annotated[Sequence[Annotated[
        _Level, BeforeValidator(TypeAdapter(InstanceOf[ScoreLevel]).validate_python),
    ]], Field(min_length=2, max_length=10)]


def _question_type(value: Any) -> str | None:
    for cls, tag in ((PredicateQuestion, 'predicate'), (ChoiceQuestion, 'choice'), (ScoreQuestion, 'score')):
        if isinstance(value, cls):
            return tag
    return None


_QuestionInput = Annotated[
    Annotated[_Question, Tag('predicate')]
    | Annotated[_ChoiceQuestion, Tag('choice')]
    | Annotated[_ScoreQuestion, Tag('score')],
    Discriminator(_question_type),
]


def _state_type(value: Any) -> str | None:
    for cls, tag in ((str, 'text'), (dict, 'object'), (list, 'array')):
        if isinstance(value, cls):
            return tag
    return None


_StateInput = Annotated[
    Annotated[str, Tag('text')]
    | Annotated[dict[str, JsonValue], Tag('object')]
    | Annotated[list[JsonValue], Tag('array')],
    Discriminator(_state_type),
]


class _Request(_Model):
    model_name: _Text
    state: _StateInput
    questions: Annotated[Sequence[_QuestionInput], Field(min_length=1)]

    @field_validator('questions')
    @classmethod
    def _questions(cls, value: Sequence[_Question], info: ValidationInfo) -> Sequence[_Question]:
        if info.context['protocol'] == 'openai' and len(value) > 400:
            raise ValueError('OpenAI accepts 1–400 questions')
        if len({question.name for question in value}) != len(value):
            raise ValueError('name: duplicate question name')
        return value


def _distribution(value: dict[_ValueT, float]) -> dict[_ValueT, float]:
    if abs(math.fsum(value.values()) - 1) > PROBABILITY_SUM_TOLERANCE:
        raise ValueError(f'sum must be 1 within {PROBABILITY_SUM_TOLERANCE}')
    return value


class _ProbabilityEntry(_Model, Generic[_ValueT]):
    value: _ValueT
    probability: _Probability


_CHOICE_ENTRIES = TypeAdapter(list[_ProbabilityEntry[str]])
_SCORE_ENTRIES = TypeAdapter(list[_ProbabilityEntry[int]])
_JEV_SCORES = TypeAdapter(dict[
    Annotated[str, StringConstraints(pattern=r'^(0|[1-9][0-9]*)$')], _Probability,
], config=ConfigDict(strict=True, allow_inf_nan=False))


def _entries(raw: Any, adapter: TypeAdapter) -> dict:
    entries = adapter.validate_python(raw, strict=True)
    result = {entry.value: entry.probability for entry in entries}
    if len(result) != len(entries):
        raise ValueError('value: duplicate distribution value')
    return result


class _Predicate(_Model):
    type: Literal['predicate']
    probability: _Probability


class _Noul(_Model):
    type: Literal['noul']
    noul: _Probability


class _Choice(_Model):
    type: Literal['choice']
    choice: str
    probabilities: Annotated[dict[str, _Probability], AfterValidator(_distribution)]
    confidence: _Probability

    @field_validator('probabilities', mode='before')
    @classmethod
    def _probabilities(cls, value: Any, info: ValidationInfo) -> Any:
        return _entries(value, _CHOICE_ENTRIES) if info.context['protocol'] == 'openai' else value


class _Score(_Model):
    type: Literal['score']
    score: Annotated[float, Field(ge=0)]
    probabilities: Annotated[dict[int, _Probability], AfterValidator(_distribution)]
    confidence: _Probability

    @field_validator('probabilities', mode='before')
    @classmethod
    def _probabilities(cls, value: Any, info: ValidationInfo) -> dict:
        if info.context['protocol'] == 'openai':
            return _entries(value, _SCORE_ENTRIES)
        return {int(k): v for k, v in _JEV_SCORES.validate_python(value).items()}


class _Refusal(_Model):
    model_config = ConfigDict(extra='allow')
    type: Literal['refusal']
    name: str | None = None
    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)


_JevAnswer = Annotated[_Noul | _Choice | _Score | _Refusal, Field(discriminator='type')]
_OpenAIAnswer = Annotated[_Predicate | _Choice | _Score | _Refusal, Field(discriminator='type')]


class _Metadata(_Model):
    model: str | None = None
    provider: str | None = None
    id: str | None = None
    usage: dict[str, JsonValue] | None = None

    @field_validator('answers', mode='before', check_fields=False)
    @classmethod
    def _answer_types(cls, value: Any, info: ValidationInfo) -> Any:
        # Check the requested type before validating its fields, so a changed
        # discriminator is diagnosed as a type mismatch rather than missing data.
        if not isinstance(value, dict):
            return value
        for name, answer in value.items():
            if isinstance(name, str) and isinstance(answer, dict):
                expected = info.context['answer_types'].get(name)
                if expected is not None and answer.get('type') not in (expected, 'refusal'):
                    raise ValueError(f'{name!r}.type: expected {expected}')
        return value


class _JevResponse(_Metadata):
    model_config = ConfigDict(title='TypeSafe/Jev')
    answers: dict[str, _JevAnswer]


class _OpenAIAnswerName(_Model):
    name: str


class _OpenAIResponse(_Metadata):
    model_config = ConfigDict(title='OpenAI')
    answers: dict[str, _OpenAIAnswer]

    @field_validator('answers', mode='before')
    @classmethod
    def _named_answers(cls, value: Any) -> dict[str, Any]:
        if not isinstance(value, list):
            raise ValueError('expected array')
        answers = {}
        for item in value:
            name = _OpenAIAnswerName.model_validate(item).name
            if name in answers:
                raise ValueError(f'name: duplicate answer name {name!r}')
            answers[name] = item
        return answers
