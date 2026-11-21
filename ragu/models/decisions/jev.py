"""Jev wire format shared by TypeSafe, OpenRouter and compatible proxies."""

from collections.abc import Sequence
from typing import Any

from ragu.models.decisions._schemas import (
    _JevResponse,
    _Noul,
    _Predicate
)
from ragu.models.decisions._validation import _build_result
from ragu.models.decisions.types import (
    ChoiceQuestion,
    DecisionQuestion,
    DecisionResult,
    PredicateQuestion,
)


def _parse_jev(
        response: Any,
        model_name: str,
        questions: Sequence[DecisionQuestion],
) -> DecisionResult:
    answer_types = {
        q.name: ('noul'
            if isinstance(q, PredicateQuestion)
            else 'choice'
            if isinstance(q, ChoiceQuestion)
            else 'score'
        )
        for q in questions
    }
    parsed = _JevResponse.model_validate(
        response,
        context={
            'protocol': 'jev',
            'answer_types': answer_types
        }
    )
    answers = {
        name: _Predicate(
            type='predicate',
            probability=answer.noul
        ) if isinstance(answer, _Noul) else answer
        for name, answer in parsed.answers.items()
    }
    return _build_result(
        model_name,
        questions,
        answers=answers,
        metadata=parsed,
        provider='TypeSafe/Jev'
    )
