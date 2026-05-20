from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Literal,
    Dict,
    Any,
    TypeVar,
    List,
    Type,
    Sequence,
    Mapping,
    Union,
    Callable,
)

from jinja2 import Environment, StrictUndefined
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
)

Role = Literal["system", "user", "assistant"]

_JINJA_ENV = Environment(
    undefined=StrictUndefined,
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)


@dataclass(frozen=True, slots=True)
class BaseMessage:
    """
    Base chat message abstraction.

    Represents a single message in a chat conversation with a fixed role
    (system, user, or assistant) and textual content. Provides conversion
    to OpenAI SDK message types.
    """
    content: str
    role: Role
    name: str | None = None

    def to_openai(self) -> ChatCompletionMessageParam:
        """
        Convert this message into a typed OpenAI ChatCompletion message.

        :return: OpenAI-compatible message payload.
        """
        if self.role == "system":
            return ChatCompletionSystemMessageParam(
                role="system",
                content=self.content,
            )

        if self.role == "user":
            return ChatCompletionUserMessageParam(
                role="user",
                content=self.content,
            )

        if self.role == "assistant":
            return ChatCompletionAssistantMessageParam(
                role="assistant",
                content=self.content,
            )

        raise ValueError(f"Unsupported role: {self.role}")

    def to_str(self) -> str:
        """
        Return a human-readable string representation of the message.

        :return: Serialized message string.
        """
        return f"[{self.role}]: {self.content}"


@dataclass(frozen=True, slots=True)
class SystemMessage(BaseMessage):
    """
    System-level instruction message.
    """
    role: Role = "system"


@dataclass(frozen=True, slots=True)
class UserMessage(BaseMessage):
    """
    User input message.
    """
    role: Role = "user"


@dataclass(frozen=True, slots=True)
class AIMessage(BaseMessage):
    """
    Assistant (LLM) response message.
    """
    role: Role = "assistant"


T = TypeVar("T", bound="ChatMessages")


@dataclass(frozen=True, slots=True)
class ChatMessages:
    """
    Container for a list of chat messages.

    Represents a single user-assistant conversation.
    """
    messages: List[BaseMessage]

    @classmethod
    def from_messages(cls: Type[T], messages: Sequence[BaseMessage]) -> T:
        """
        Construct a ChatMessages instance from a sequence of messages.

        :param messages: Source message sequence.
        :return: ChatMessages container.
        """
        return cls(messages=list(messages))

    def to_openai(self) -> List[ChatCompletionMessageParam]:
        """
        Convert all messages to OpenAI ChatCompletion message parameters.

        :return: List of OpenAI-compatible message payloads.
        """
        return [m.to_openai() for m in self.messages]

    def __iter__(self):
        return iter(self.messages)

    def __len__(self):
        return len(self.messages)

    def to_str(self) -> str:
        """
        Return a readable multi-line string representation of the conversation.

        :return: Multi-line serialized conversation.
        """
        return "\n".join([m.to_str() for m in self.messages])


def render(template_conversation: Union[BaseMessage, ChatMessages], **params: Any) -> List[ChatMessages]:
    """
    Render Jinja2 templates inside message contents in batch mode.

    :param template_conversation: Message or conversation template.
    :param params: Scalar and batch Jinja context parameters.
    :return: Rendered conversations (batch size inferred from list/tuple params).
    :raises ValueError: If batch parameter lengths are inconsistent.
    """

    def _is_batch_value(v: Any) -> bool:
        return isinstance(v, (list, tuple))

    def _infer_batch_size(params: Mapping[str, Any]) -> int:
        sizes = {len(v) for v in params.values() if _is_batch_value(v)}
        if not sizes:
            return 1
        if len(sizes) != 1:
            raise ValueError(f"Batch parameters have different sizes: {sorted(sizes)}.")
        return next(iter(sizes))

    def _build_row_context(params: Mapping[str, Any], i: int) -> Dict[str, Any]:
        row: Dict[str, Any] = {}
        for k, v in params.items():
            row[k] = v[i] if _is_batch_value(v) else v
        return row

    if isinstance(template_conversation, BaseMessage):
        template_cm = ChatMessages.from_messages([template_conversation])
    else:
        template_cm = template_conversation

    n = _infer_batch_size(params)

    for k, v in params.items():
        if _is_batch_value(v) and len(v) != n:
            raise ValueError(
                f"Batch parameter '{k}' has length {len(v)}, expected {n}."
            )

    out: List[ChatMessages] = []
    for i in range(n):
        ctx = _build_row_context(params, i)

        rendered_msgs: List[BaseMessage] = []
        for m in template_cm.messages:
            tmpl = _JINJA_ENV.from_string(m.content)
            new_content = tmpl.render(**ctx)

            msg_type = type(m)
            rendered_msgs.append(
                msg_type(role=m.role, content=new_content)
            )

        out.append(ChatMessages.from_messages(rendered_msgs))

    return out


if TYPE_CHECKING:
    from ragu.common.prompts.few_shot import FewShotFormatter


def render_with_few_shots(
    template_conversation: Union[BaseMessage, ChatMessages],
    examples_list: List[List[dict[str, Any]] | None],
    few_shot_formatter: FewShotFormatter | None = None,
    **params: Any,
) -> List[ChatMessages]:
    conversations = render(template_conversation, **params)

    if few_shot_formatter is None or not any(examples_list):
        return conversations

    result: List[ChatMessages] = []
    for conv, examples in zip(conversations, examples_list):
        if not examples:
            result.append(conv)
            continue

        few_shot_pairs: List[BaseMessage] = []
        for example in examples:
            user_msg, ai_msg = few_shot_formatter(example)
            few_shot_pairs.append(user_msg)
            few_shot_pairs.append(ai_msg)

        new_messages = list(conv.messages[:-1]) + few_shot_pairs + [conv.messages[-1]]
        result.append(ChatMessages.from_messages(new_messages))

    return result
