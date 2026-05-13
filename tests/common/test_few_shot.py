import json

import pytest

from ragu.common.prompts.few_shot import (
    format_artifact_extraction_example,
    format_artifact_validation_example,
    format_entity_extraction_example,
    format_entity_validation_example,
    format_relation_extraction_example,
    format_relation_validation_example,
)
from ragu.common.prompts.messages import (
    AIMessage,
    ChatMessages,
    SystemMessage,
    UserMessage,
    render_with_few_shots,
)
from ragu.common.prompts.prompt_storage import DEFAULT_PROMPT_TEMPLATES, RAGUInstruction
from ragu.triplet.prompts import (
    TWO_STAGE_ENTITY_EXTRACTION_INSTRUCTION,
    TWO_STAGE_ENTITY_VALIDATION_INSTRUCTION,
    TWO_STAGE_RELATION_EXTRACTION_INSTRUCTION,
    TWO_STAGE_RELATION_VALIDATION_INSTRUCTION,
)


def _make_entity(name: str = "Test", etype: str = "PERSON", desc: str = "desc"):
    return {
        "entity_name": name,
        "entity_type": etype,
        "description": desc,
    }


def _make_relation(
    src: str = "A",
    tgt: str = "B",
    rtype: str = "RELATED_TO",
    strength: int = 3,
    desc: str = "desc",
):
    return {
        "source_entity": src,
        "target_entity": tgt,
        "relation_type": rtype,
        "relationship_strength": strength,
        "description": desc,
    }


def _make_example(
    input_text: str = "Test text",
    entities: list | None = None,
    relations: list | None = None,
):
    output: dict = {}
    if entities is not None:
        output["entities"] = entities
    if relations is not None:
        output["relations"] = relations
    return {
        "id": "test-id",
        "input_text": input_text,
        "output": output,
        "metadata": {},
        "language": "english",
    }


class TestFormatters:
    def test_artifact_extraction_formatter_roles(self):
        example = _make_example(
            entities=[_make_entity()],
            relations=[_make_relation()],
        )
        user_msg, ai_msg = format_artifact_extraction_example(example)
        assert isinstance(user_msg, UserMessage)
        assert isinstance(ai_msg, AIMessage)

    def test_artifact_extraction_formatter_content(self):
        example = _make_example(
            entities=[_make_entity("Apple", "ORG")],
            relations=[_make_relation()],
        )
        user_msg, ai_msg = format_artifact_extraction_example(example)
        assert "Test text" in user_msg.content
        parsed = json.loads(ai_msg.content)
        assert "entities" in parsed
        assert "relations" in parsed

    def test_artifact_validation_formatter(self):
        example = _make_example(
            entities=[_make_entity("Apple", "ORG")],
            relations=[_make_relation("Apple", "Google")],
        )
        user_msg, ai_msg = format_artifact_validation_example(example)
        assert "Triplets for validation:" in user_msg.content
        assert "Text for validation:" in user_msg.content
        assert "Apple" in user_msg.content
        parsed = json.loads(ai_msg.content)
        assert "entities" in parsed
        assert "relations" in parsed

    def test_entity_extraction_formatter(self):
        example = _make_example(entities=[_make_entity("Einstein", "PERSON")])
        user_msg, ai_msg = format_entity_extraction_example(example)
        assert user_msg.content.startswith("Text:\n")
        parsed = json.loads(ai_msg.content)
        assert "entities" in parsed
        assert "relations" not in parsed
        assert parsed["entities"][0]["entity_name"] == "Einstein"

    def test_entity_validation_formatter(self):
        example = _make_example(
            entities=[_make_entity("Einstein", "PERSON", "A physicist")]
        )
        user_msg, ai_msg = format_entity_validation_example(example)
        assert "Entities for validation:" in user_msg.content
        assert "Einstein" in user_msg.content
        parsed = json.loads(ai_msg.content)
        assert "entities" in parsed

    def test_relation_extraction_formatter(self):
        example = _make_example(relations=[_make_relation("A", "B", "KNOWS")])
        user_msg, ai_msg = format_relation_extraction_example(example)
        assert user_msg.content.startswith("Text:\n")
        parsed = json.loads(ai_msg.content)
        assert "relations" in parsed
        assert "entities" not in parsed

    def test_relation_validation_formatter(self):
        example = _make_example(
            relations=[_make_relation("A", "B", "KNOWS", 4, "they know each other")]
        )
        user_msg, ai_msg = format_relation_validation_example(example)
        assert "Relations for validation:" in user_msg.content
        assert "A" in user_msg.content
        parsed = json.loads(ai_msg.content)
        assert "relations" in parsed

    def test_entity_validation_formatter_empty_entities(self):
        example = _make_example(entities=[])
        user_msg, ai_msg = format_entity_validation_example(example)
        assert "(none)" in user_msg.content

    def test_relation_validation_formatter_empty_relations(self):
        example = _make_example(relations=[])
        user_msg, ai_msg = format_relation_validation_example(example)
        assert "(none)" in user_msg.content


class TestRenderWithFewShots:
    def test_no_examples_returns_render_result(self):
        template = ChatMessages.from_messages([
            SystemMessage(content="Instructions"),
            UserMessage(content="Text: {{ context }}"),
        ])
        result = render_with_few_shots(
            template,
            examples_list=[None],
            few_shot_formatter=format_entity_extraction_example,
            context=["Hello world"],
        )
        assert len(result) == 1
        assert len(result[0].messages) == 2
        assert result[0].messages[0].role == "system"
        assert result[0].messages[1].role == "user"

    def test_no_formatter_returns_render_result(self):
        template = ChatMessages.from_messages([
            SystemMessage(content="Instructions"),
            UserMessage(content="Text: {{ context }}"),
        ])
        examples = [_make_example(entities=[_make_entity()])]
        result = render_with_few_shots(
            template,
            examples_list=[examples],
            few_shot_formatter=None,
            context=["Hello world"],
        )
        assert len(result) == 1
        assert len(result[0].messages) == 2

    def test_with_examples_inserts_pairs(self):
        template = ChatMessages.from_messages([
            SystemMessage(content="Instructions"),
            UserMessage(content="Text: {{ context }}"),
        ])
        ex1 = _make_example(entities=[_make_entity("A")])
        ex2 = _make_example(entities=[_make_entity("B")])
        result = render_with_few_shots(
            template,
            examples_list=[[ex1, ex2]],
            few_shot_formatter=format_entity_extraction_example,
            context=["Hello world"],
        )
        assert len(result) == 1
        msgs = result[0].messages
        assert len(msgs) == 6  # system + user/ai + user/ai + user(task)
        assert msgs[0].role == "system"
        assert msgs[1].role == "user"
        assert msgs[2].role == "assistant"
        assert msgs[3].role == "user"
        assert msgs[4].role == "assistant"
        assert msgs[5].role == "user"
        assert "Hello world" in msgs[5].content

    def test_batch_with_mixed_examples(self):
        template = ChatMessages.from_messages([
            SystemMessage(content="Instructions"),
            UserMessage(content="Text: {{ context }}"),
        ])
        ex1 = _make_example(entities=[_make_entity("A")])
        result = render_with_few_shots(
            template,
            examples_list=[[ex1], None],
            few_shot_formatter=format_entity_extraction_example,
            context=["Chunk 1", "Chunk 2"],
        )
        assert len(result) == 2
        assert len(result[0].messages) == 4  # system + user/ai + user
        assert len(result[1].messages) == 2  # system + user

    def test_empty_examples_list(self):
        template = ChatMessages.from_messages([
            SystemMessage(content="Instructions"),
            UserMessage(content="Text: {{ context }}"),
        ])
        result = render_with_few_shots(
            template,
            examples_list=[[]],
            few_shot_formatter=format_entity_extraction_example,
            context=["Hello"],
        )
        assert len(result) == 1
        assert len(result[0].messages) == 2

    def test_to_openai_round_trip(self):
        template = ChatMessages.from_messages([
            SystemMessage(content="Instructions"),
            UserMessage(content="Text: {{ context }}"),
        ])
        ex = _make_example(entities=[_make_entity("Test")])
        result = render_with_few_shots(
            template,
            examples_list=[[ex]],
            few_shot_formatter=format_entity_extraction_example,
            context=["Hello"],
        )
        openai_msgs = result[0].to_openai()
        assert openai_msgs[0]["role"] == "system"
        assert openai_msgs[1]["role"] == "user"
        assert openai_msgs[2]["role"] == "assistant"
        assert openai_msgs[3]["role"] == "user"


class TestInstructionFormatters:
    def test_artifact_extraction_has_formatter(self):
        instruction = DEFAULT_PROMPT_TEMPLATES["artifact_extraction"]
        assert instruction.few_shot_formatter is not None
        assert instruction.few_shot_formatter is format_artifact_extraction_example

    def test_artifact_validation_has_formatter(self):
        instruction = DEFAULT_PROMPT_TEMPLATES["artifact_validation"]
        assert instruction.few_shot_formatter is not None
        assert instruction.few_shot_formatter is format_artifact_validation_example

    def test_two_stage_entity_extraction_has_formatter(self):
        assert TWO_STAGE_ENTITY_EXTRACTION_INSTRUCTION.few_shot_formatter is not None
        assert (
            TWO_STAGE_ENTITY_EXTRACTION_INSTRUCTION.few_shot_formatter
            is format_entity_extraction_example
        )

    def test_two_stage_entity_validation_has_formatter(self):
        assert TWO_STAGE_ENTITY_VALIDATION_INSTRUCTION.few_shot_formatter is not None
        assert (
            TWO_STAGE_ENTITY_VALIDATION_INSTRUCTION.few_shot_formatter
            is format_entity_validation_example
        )

    def test_two_stage_relation_extraction_has_formatter(self):
        assert TWO_STAGE_RELATION_EXTRACTION_INSTRUCTION.few_shot_formatter is not None
        assert (
            TWO_STAGE_RELATION_EXTRACTION_INSTRUCTION.few_shot_formatter
            is format_relation_extraction_example
        )

    def test_two_stage_relation_validation_has_formatter(self):
        assert TWO_STAGE_RELATION_VALIDATION_INSTRUCTION.few_shot_formatter is not None
        assert (
            TWO_STAGE_RELATION_VALIDATION_INSTRUCTION.few_shot_formatter
            is format_relation_validation_example
        )

    def test_icl_instructions_use_system_message(self):
        for name in ("artifact_extraction", "artifact_validation"):
            instruction = DEFAULT_PROMPT_TEMPLATES[name]
            assert instruction.messages.messages[0].role == "system"
            assert instruction.messages.messages[-1].role == "user"

    def test_two_stage_instructions_use_system_message(self):
        for instruction in (
            TWO_STAGE_ENTITY_EXTRACTION_INSTRUCTION,
            TWO_STAGE_ENTITY_VALIDATION_INSTRUCTION,
            TWO_STAGE_RELATION_EXTRACTION_INSTRUCTION,
            TWO_STAGE_RELATION_VALIDATION_INSTRUCTION,
        ):
            assert instruction.messages.messages[0].role == "system"
            assert instruction.messages.messages[-1].role == "user"

    def test_non_icl_instructions_unchanged(self):
        for name in ("community_report", "entity_summarizer", "relation_summarizer"):
            instruction = DEFAULT_PROMPT_TEMPLATES[name]
            assert instruction.few_shot_formatter is None
            assert len(instruction.messages.messages) == 1
            assert instruction.messages.messages[0].role == "user"
