"""OpenAI Decisions wire conversion, independent of the SDK transport."""

from collections.abc import Sequence
from typing import Any

from ragu.models.decisions._schemas import _OpenAIResponse
from ragu.models.decisions._validation import _build_result
from ragu.models.decisions.types import (
    ChoiceQuestion,
    DecisionQuestion,
    DecisionResult,
    PredicateQuestion,
)


def _parse_openai(
        response: Any,
        model_name: str,
        questions: Sequence[DecisionQuestion],
) -> DecisionResult:
    answer_types = {
        q.name: (
            'predicate' if isinstance(q, PredicateQuestion)
            else 'choice' if isinstance(q, ChoiceQuestion) else 'score'
        )
        for q in questions
    }
    parsed = _OpenAIResponse.model_validate(
        response,
        context={'protocol': 'openai', 'answer_types': answer_types},
    )
    return _build_result(
        model_name,
        questions,
        answers=parsed.answers,
        metadata=parsed,
        provider='OpenAI'
    )
