from __future__ import annotations

import json
from typing import Any, Callable

from ragu.common.prompts.messages import AIMessage, UserMessage

FewShotFormatter = Callable[[dict[str, Any]], tuple[UserMessage, AIMessage]]


def _format_entities(entities: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for e in entities:
        name = e.get("entity_name", "")
        etype = e.get("entity_type", "")
        desc = e.get("description", "")
        parts.append(f"- {name} ({etype}): {desc}")
    return "\n".join(parts)


def _format_relations(relations: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for r in relations:
        src = r.get("source_entity", "")
        tgt = r.get("target_entity", "")
        rtype = r.get("relation_type", "")
        strength = r.get("relationship_strength", "")
        desc = r.get("description", "")
        parts.append(f"- {src} → {tgt} ({rtype}, strength: {strength}): {desc}")
    return "\n".join(parts)


def _serialize_output(output: dict[str, Any]) -> str:
    return json.dumps(output, ensure_ascii=False, indent=2)


def format_artifact_extraction_example(
    example: dict[str, Any],
) -> tuple[UserMessage, AIMessage]:
    user_content = f"Text:\n{example['input_text']}"
    ai_content = _serialize_output(example["output"])
    return UserMessage(content=user_content), AIMessage(content=ai_content)


def format_artifact_validation_example(
    example: dict[str, Any],
) -> tuple[UserMessage, AIMessage]:
    input_text = example["input_text"]
    output = example["output"]
    entities = output.get("entities", [])
    relations = output.get("relations", [])

    parts: list[str] = []
    if entities:
        parts.append(f"Entities:\n{_format_entities(entities)}")
    if relations:
        parts.append(f"Relations:\n{_format_relations(relations)}")

    artifacts_block = "\n\n".join(parts)
    user_content = (
        f"Triplets for validation:\n{artifacts_block}\n\n"
        f"Text for validation:\n{input_text}"
    )
    ai_content = _serialize_output(output)
    return UserMessage(content=user_content), AIMessage(content=ai_content)


def format_entity_extraction_example(
    example: dict[str, Any],
) -> tuple[UserMessage, AIMessage]:
    user_content = f"Text:\n{example['input_text']}"
    output = {"entities": example["output"].get("entities", [])}
    ai_content = _serialize_output(output)
    return UserMessage(content=user_content), AIMessage(content=ai_content)


def format_entity_validation_example(
    example: dict[str, Any],
) -> tuple[UserMessage, AIMessage]:
    input_text = example["input_text"]
    output = example["output"]
    entities = output.get("entities", [])

    entities_block = _format_entities(entities) if entities else "(none)"
    user_content = (
        f"Entities for validation:\n{entities_block}\n\n"
        f"Text:\n{input_text}"
    )
    ai_content = _serialize_output(output)
    return UserMessage(content=user_content), AIMessage(content=ai_content)


def format_relation_extraction_example(
    example: dict[str, Any],
) -> tuple[UserMessage, AIMessage]:
    user_content = f"Text:\n{example['input_text']}"
    output = {"relations": example["output"].get("relations", [])}
    ai_content = _serialize_output(output)
    return UserMessage(content=user_content), AIMessage(content=ai_content)


def format_relation_validation_example(
    example: dict[str, Any],
) -> tuple[UserMessage, AIMessage]:
    input_text = example["input_text"]
    output = example["output"]
    relations = output.get("relations", [])

    relations_block = _format_relations(relations) if relations else "(none)"
    user_content = (
        f"Relations for validation:\n{relations_block}\n\n"
        f"Text:\n{input_text}"
    )
    ai_content = _serialize_output(output)
    return UserMessage(content=user_content), AIMessage(content=ai_content)
