from __future__ import annotations

from dataclasses import dataclass
from typing import Type, TypeVar

from pydantic import BaseModel

from ragu.common.prompts.default_models import (
    ArtifactsModel,
    CommunityReportModel,
    GlobalSearchContextModel,
    EntityDescriptionModel,
    RelationDescriptionModel,
    ClusterSummarizationModel,
    QueryPlan,
    RewriteQuery,
)
from ragu.common.prompts.default_templates import (
    DEFAULT_ARTIFACTS_EXTRACTOR_SYSTEM,
    DEFAULT_ARTIFACTS_EXTRACTOR_USER,
    DEFAULT_ARTIFACTS_VALIDATOR_SYSTEM,
    DEFAULT_ARTIFACTS_VALIDATOR_USER,
    DEFAULT_COMMUNITY_REPORT_PROMPT,
    DEFAULT_RELATIONSHIP_SUMMARIZER_PROMPT,
    DEFAULT_ENTITY_SUMMARIZER_PROMPT,
    DEFAULT_RESPONSE_ONLY_PROMPT,
    DEFAULT_MIX_SEARCH_PROMPT,
    DEFAULT_MIX_SEARCH_CONTEXT_PROMPT,
    DEFAULT_GLOBAL_SEARCH_CONTEXT_PROMPT,
    DEFAULT_GLOBAL_SEARCH_PROMPT,
    DEFAULT_CLUSTER_SUMMARIZER_PROMPT,
    DEFAULT_RAGU_LM_ENTITY_EXTRACTION_PROMPT,
    DEFAULT_RAGU_LM_ENTITY_NORMALIZATION_PROMPT,
    DEFAULT_RAGU_LM_ENTITY_DESCRIPTION_PROMPT,
    DEFAULT_RAGU_LM_RELATION_DESCRIPTION_PROMPT,
    DEFAULT_QUERY_DECOMPOSITION_PROMPT,
    DEFAULT_QUERY_REWRITE_PROMPT,
    DEFAULT_RAGU_LM_SYSTEM_PROMPT,
)
from ragu.common.prompts.few_shot import (
    FewShotFormatter,
    format_artifact_extraction_example,
    format_artifact_validation_example,
)
from ragu.common.prompts.messages import (
    ChatMessages,
    UserMessage,
    SystemMessage,
)


#: Concrete output schema a component requires from its prompt.
ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class RAGUInstruction:
    messages: ChatMessages
    pydantic_model: Type[BaseModel] | Type[str] = str
    description: str | None = None
    few_shot_formatter: FewShotFormatter | None = None


def require_prompt_schema(
    instruction: RAGUInstruction,
    prompt_name: str,
    expected_model: Type[ModelT],
) -> Type[ModelT]:
    """
    Validate that a prompt instruction uses the output schema a component expects.

    Returns the schema so that callers can pass it straight to
    :meth:`~ragu.models.llm.LLM.chat_completion`: ``instruction.pydantic_model``
    is only known as ``Type[BaseModel] | Type[str]``, while the validated model
    is a concrete class and carries through to the response type.

    :param instruction: Prompt instruction to check.
    :type instruction: RAGUInstruction
    :param prompt_name: Prompt name used in the error message.
    :type prompt_name: str
    :param expected_model: Output schema type the calling component requires.
    :type expected_model: Type[ModelT]
    :returns: The same ``expected_model``, once validated.
    :rtype: Type[ModelT]
    :raises ValueError: If the instruction's ``pydantic_model`` differs from
        ``expected_model`` (e.g. the prompt was replaced with an incompatible one).
    """
    if instruction.pydantic_model is not expected_model:
        actual = getattr(
            instruction.pydantic_model, "__name__", repr(instruction.pydantic_model)
        )
        raise ValueError(
            f"Prompt {prompt_name!r} must use output schema "
            f"{expected_model.__name__}, got {actual}. The prompt template was "
            f"likely replaced with an incompatible schema via update_prompt()."
        )
    return expected_model


DEFAULT_PROMPT_TEMPLATES: dict[str, RAGUInstruction] = {
    "artifact_extraction": RAGUInstruction(
        messages=ChatMessages.from_messages(
            [
                SystemMessage(content=DEFAULT_ARTIFACTS_EXTRACTOR_SYSTEM),
                UserMessage(content=DEFAULT_ARTIFACTS_EXTRACTOR_USER),
            ]
        ),
        pydantic_model=ArtifactsModel,
        description="Prompt for extracting artifacts (entities and relations) from a text passage.",
        few_shot_formatter=format_artifact_extraction_example,
    ),

    "artifact_validation": RAGUInstruction(
        messages=ChatMessages.from_messages(
            [
                SystemMessage(content=DEFAULT_ARTIFACTS_VALIDATOR_SYSTEM),
                UserMessage(content=DEFAULT_ARTIFACTS_VALIDATOR_USER),
            ]
        ),
        pydantic_model=ArtifactsModel,
        description="Prompt for validating extracted artifacts against a schema.",
        few_shot_formatter=format_artifact_validation_example,
    ),

    "community_report": RAGUInstruction(
        messages=ChatMessages.from_messages(
            [
                UserMessage(content=DEFAULT_COMMUNITY_REPORT_PROMPT),
            ]
        ),
        pydantic_model=CommunityReportModel,
        description="Prompt for generating community summaries from contextual data.",
    ),

    "entity_summarizer": RAGUInstruction(
        messages=ChatMessages.from_messages(
            [
                UserMessage(content=DEFAULT_ENTITY_SUMMARIZER_PROMPT),
            ]
        ),
        pydantic_model=EntityDescriptionModel,
        description="Prompt for summarizing entity descriptions.",
    ),

    "relation_summarizer": RAGUInstruction(
        messages=ChatMessages.from_messages(
            [
                UserMessage(content=DEFAULT_RELATIONSHIP_SUMMARIZER_PROMPT),
            ]
        ),
        pydantic_model=RelationDescriptionModel,
        description="Prompt for summarizing relationship descriptions.",
    ),

    "global_search_context": RAGUInstruction(
        messages=ChatMessages.from_messages(
            [
                UserMessage(content=DEFAULT_GLOBAL_SEARCH_CONTEXT_PROMPT),
            ]
        ),
        pydantic_model=GlobalSearchContextModel,
        description="Prompt for generating contextual information for a global search.",
    ),

    "global_search": RAGUInstruction(
        messages=ChatMessages.from_messages(
            [
                UserMessage(content=DEFAULT_GLOBAL_SEARCH_PROMPT),
            ]
        ),
        pydantic_model=str,
        description="Prompt for generating a synthesized global search response.",
    ),

    "local_search": RAGUInstruction(
        messages=ChatMessages.from_messages(
            [
                UserMessage(content=DEFAULT_RESPONSE_ONLY_PROMPT),
            ]
        ),
        pydantic_model=str,
        description="Prompt for generating a local context-based search response.",
    ),

    "naive_search": RAGUInstruction(
        messages=ChatMessages.from_messages(
            [
                UserMessage(content=DEFAULT_RESPONSE_ONLY_PROMPT),
            ]
        ),
        pydantic_model=str,
        description="Prompt for generating a naive vector RAG search response.",
    ),

    "mix_search": RAGUInstruction(
        messages=ChatMessages.from_messages(
            [
                UserMessage(content=DEFAULT_MIX_SEARCH_PROMPT),
            ]
        ),
        pydantic_model=str,
        description="Prompt for generating an ensemble response from multiple search contexts.",
    ),

    "mix_search_context": RAGUInstruction(
        messages=ChatMessages.from_messages(
            [
                UserMessage(content=DEFAULT_MIX_SEARCH_CONTEXT_PROMPT),
            ]
        ),
        pydantic_model=str,
        description="Prompt for formatting ordered multi-engine contexts before final synthesis.",
    ),

    "cluster_summarize": RAGUInstruction(
        messages=ChatMessages.from_messages(
            [
                UserMessage(content=DEFAULT_CLUSTER_SUMMARIZER_PROMPT),
            ]
        ),
        pydantic_model=ClusterSummarizationModel,
        description=None,
    ),

    "ragu_lm_entity_extraction": RAGUInstruction(
        messages=ChatMessages.from_messages(
            [
                SystemMessage(content=DEFAULT_RAGU_LM_SYSTEM_PROMPT),
                UserMessage(content=DEFAULT_RAGU_LM_ENTITY_EXTRACTION_PROMPT),
            ]
        ),
        pydantic_model=str,
        description="Instruction for RAGU-lm entity extraction stage.",
    ),

    "ragu_lm_entity_normalization": RAGUInstruction(
        messages=ChatMessages.from_messages(
            [
                SystemMessage(content=DEFAULT_RAGU_LM_SYSTEM_PROMPT),
                UserMessage(content=DEFAULT_RAGU_LM_ENTITY_NORMALIZATION_PROMPT),
            ]
        ),
        pydantic_model=str,
        description="Instruction for RAGU-lm entity normalization stage.",
    ),

    "ragu_lm_entity_description": RAGUInstruction(
        messages=ChatMessages.from_messages(
            [
                SystemMessage(content=DEFAULT_RAGU_LM_SYSTEM_PROMPT),
                UserMessage(content=DEFAULT_RAGU_LM_ENTITY_DESCRIPTION_PROMPT),
            ]
        ),
        pydantic_model=str,
        description="Instruction for RAGU-lm entity description stage.",
    ),

    "ragu_lm_relation_description": RAGUInstruction(
        messages=ChatMessages.from_messages(
            [
                SystemMessage(content=DEFAULT_RAGU_LM_SYSTEM_PROMPT),
                UserMessage(content=DEFAULT_RAGU_LM_RELATION_DESCRIPTION_PROMPT),
            ]
        ),
        pydantic_model=str,
        description="Instruction for RAGU-lm relation description stage.",
    ),

    "query_decomposition": RAGUInstruction(
        messages=ChatMessages.from_messages(
            [
                UserMessage(content=DEFAULT_QUERY_DECOMPOSITION_PROMPT),
            ]
        ),
        pydantic_model=QueryPlan,
        description="Prompt for decomposing a complex query into atomic subqueries with dependencies.",
    ),

    "query_rewrite": RAGUInstruction(
        messages=ChatMessages.from_messages(
            [
                UserMessage(content=DEFAULT_QUERY_REWRITE_PROMPT),
            ]
        ),
        pydantic_model=RewriteQuery,
        description="Prompt for rewriting a subquery using answers from its dependencies.",
    ),
}
