"""Assemble public decision results from validated answers."""

from collections.abc import Mapping, Sequence

from ragu.models.decisions._schemas import (
    _Choice,
    _Metadata,
    _Predicate,
    _Refusal,
    _Score,
)
from ragu.models.decisions.types import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionQuestion,
    DecisionResult,
    PredicateAnswer,
    RefusalAnswer,
    ScoreAnswer,
    ScoreQuestion,
)


def _build_result(
    model_name: str,
    questions: Sequence[DecisionQuestion],
    *,
    answers: Mapping[str, _Predicate | _Choice | _Score | _Refusal],
    metadata: _Metadata,
    provider: str,
) -> DecisionResult:
    """Build the public result from validated, normalized answers and metadata."""
    expected = {question.name for question in questions}
    if set(answers) != expected:
        missing, extra = expected - set(answers), set(answers) - expected
        raise ValueError(f'{provider} answers.name: missing {sorted(missing)}, unexpected {sorted(extra)}')

    answer_classes = {
        'predicate': PredicateAnswer,
        'choice': ChoiceAnswer,
        'score': ScoreAnswer,
        'refusal': RefusalAnswer,
    }
    parsed = {}
    for question in questions:
        answer = answers[question.name]
        fields = (
            {'details': answer.model_extra or None}
            if isinstance(answer, _Refusal)
            else answer.model_dump(exclude={'type'})
        )
        if isinstance(answer, (_Choice, _Score)):
            assert isinstance(question, (ChoiceQuestion, ScoreQuestion))
            keys = (
                [option.value for option in question.choices]
                if isinstance(question, ChoiceQuestion)
                else list(range(len(question.levels)))
            )
            field = f'{provider} answers[{question.name!r}]'
            if isinstance(answer, _Choice) and answer.choice not in keys:
                raise ValueError(f'{field}.choice: unknown value')
            if isinstance(answer, _Score) and answer.score > keys[-1]:
                raise ValueError(f'{field}.score: expected number in [0, {keys[-1]}]')
            if set(answer.probabilities) != set(keys):
                raise ValueError(f'{field}.probabilities: expected all and only requested values')
            fields['probabilities'] = {key: answer.probabilities[key] for key in keys}
        parsed[question.name] = answer_classes[answer.type](**fields)

    return DecisionResult(
        answers=parsed,
        requested_model=model_name,
        provider=metadata.provider,
        resolved_model=metadata.model,
        id=metadata.id,
        usage=metadata.usage,
    )
