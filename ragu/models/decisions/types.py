"""Transport-independent decision questions and results."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

_JSONValue: TypeAlias = str | int | float | bool | None | list['_JSONValue'] | dict[str, '_JSONValue']
DecisionState: TypeAlias = str | dict[str, _JSONValue] | list[_JSONValue]


@dataclass(frozen=True, slots=True)
class PredicateQuestion:
    name: str
    instructions: str


@dataclass(frozen=True, slots=True)
class ChoiceOption:
    value: str
    description: str


@dataclass(frozen=True, slots=True)
class ChoiceQuestion:
    name: str
    instructions: str
    choices: Sequence[ChoiceOption]


@dataclass(frozen=True, slots=True)
class ScoreLevel:
    label: str
    description: str


@dataclass(frozen=True, slots=True)
class ScoreQuestion:
    name: str
    instructions: str
    levels: Sequence[ScoreLevel]


DecisionQuestion: TypeAlias = PredicateQuestion | ChoiceQuestion | ScoreQuestion


@dataclass(slots=True)
class PredicateAnswer:
    probability: float


@dataclass(slots=True)
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float


@dataclass(slots=True)
class ScoreAnswer:
    score: float
    probabilities: dict[int, float]
    confidence: float


@dataclass(slots=True)
class RefusalAnswer:
    details: Any | None = None


DecisionAnswer: TypeAlias = PredicateAnswer | ChoiceAnswer | ScoreAnswer | RefusalAnswer


@dataclass(slots=True)
class DecisionResult:
    answers: dict[str, DecisionAnswer]
    requested_model: str
    provider: str | None = None
    resolved_model: str | None = None
    id: str | None = None
    usage: dict[str, Any] | None = None
    cache_hit: bool = False


class DecisionClient(Protocol):
    async def decide(
        self, model_name: str, *, state: DecisionState, questions: Sequence[DecisionQuestion],
    ) -> DecisionResult:
        """Evaluate named questions about one state.

        :param model_name: Model identifier understood by the configured service.
        :param state: Text or structured JSON context.
        :param questions: Ordered, uniquely named questions.
        :return: Validated answers in question order, with available metadata.
        """
        ...
