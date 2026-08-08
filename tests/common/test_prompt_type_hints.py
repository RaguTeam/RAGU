from ragu.common.prompts.messages import render
from ragu.common.prompts.prompt_storage import DEFAULT_PROMPT_TEMPLATES
from ragu.triplet.prompts import (
    TWO_STAGE_ENTITY_VALIDATION_INSTRUCTION,
    TWO_STAGE_RELATION_EXTRACTION_INSTRUCTION,
    TWO_STAGE_RELATION_VALIDATION_INSTRUCTION,
)

ENTITY_TYPES = "PERSON, ORGANIZATION"
RELATION_TYPES = "WORKPLACE, LOCATED_IN"


def _render_single(instruction, **params):
    conversations = render(instruction.messages, **params)
    assert len(conversations) == 1
    return "\n".join(message.content for message in conversations[0].messages)


def test_artifact_extraction_prompt_carries_both_type_lists():
    rendered = _render_single(
        DEFAULT_PROMPT_TEMPLATES["artifact_extraction"],
        context=["some text"],
        language="English",
        entity_types=ENTITY_TYPES,
        relation_types=RELATION_TYPES,
    )

    assert ENTITY_TYPES in rendered
    assert RELATION_TYPES in rendered


def test_artifact_validation_prompt_carries_both_type_lists():
    rendered = _render_single(
        DEFAULT_PROMPT_TEMPLATES["artifact_validation"],
        context=["some text"],
        artifacts=["stub artifacts"],
        language="English",
        entity_types=ENTITY_TYPES,
        relation_types=RELATION_TYPES,
    )

    assert ENTITY_TYPES in rendered
    assert RELATION_TYPES in rendered


def test_artifact_validation_prompt_omits_type_lists_when_not_set():
    rendered = _render_single(
        DEFAULT_PROMPT_TEMPLATES["artifact_validation"],
        context=["some text"],
        artifacts=["stub artifacts"],
        language="English",
        entity_types=None,
        relation_types=None,
    )

    assert "must be one of the following" not in rendered


def test_signature_legend_appears_only_with_signatures():
    for instruction, extra in [
        (DEFAULT_PROMPT_TEMPLATES["artifact_extraction"], {}),
        (DEFAULT_PROMPT_TEMPLATES["artifact_validation"], {"artifacts": ["stub"]}),
        (TWO_STAGE_RELATION_EXTRACTION_INSTRUCTION, {"entities": [[]]}),
        (TWO_STAGE_RELATION_VALIDATION_INSTRUCTION, {"entities": [[]], "relations": [[]]}),
    ]:
        common = dict(
            context=["t"], language="English",
            entity_types=ENTITY_TYPES, relation_types=RELATION_TYPES, **extra,
        )

        assert "TYPE [SUBJECT -> OBJECT]" in _render_single(
            instruction, type_signatures=True, **common
        )
        assert "TYPE [SUBJECT -> OBJECT]" not in _render_single(
            instruction, type_signatures=False, **common
        )
        # The variable was added later: templates must still render without it.
        assert "TYPE [SUBJECT -> OBJECT]" not in _render_single(instruction, **common)


def test_legend_is_omitted_when_there_are_no_relation_types():
    rendered = _render_single(
        TWO_STAGE_RELATION_EXTRACTION_INSTRUCTION,
        context=["t"], entities=[[]], language="English",
        entity_types=None, relation_types=None, type_signatures=True,
    )

    assert "TYPE [SUBJECT -> OBJECT]" not in rendered


def test_two_stage_validation_prompts_carry_their_type_lists():
    entity_rendered = _render_single(
        TWO_STAGE_ENTITY_VALIDATION_INSTRUCTION,
        context=["some text"],
        entities=[[]],
        language="English",
        entity_types=ENTITY_TYPES,
    )
    relation_rendered = _render_single(
        TWO_STAGE_RELATION_VALIDATION_INSTRUCTION,
        context=["some text"],
        entities=[[]],
        relations=[[]],
        language="English",
        relation_types=RELATION_TYPES,
    )

    assert ENTITY_TYPES in entity_rendered
    assert RELATION_TYPES in relation_rendered
