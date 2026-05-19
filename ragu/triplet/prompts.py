from __future__ import annotations

from ragu.common.prompts.default_models import (
    EntitiesExtractionModel,
    RelationsExtractionModel,
)
from ragu.common.prompts.few_shot import (
    format_entity_extraction_example,
    format_entity_validation_example,
    format_relation_extraction_example,
    format_relation_validation_example,
)
from ragu.common.prompts.messages import ChatMessages, SystemMessage, UserMessage
from ragu.common.prompts.prompt_storage import RAGUInstruction


DEFAULT_TWO_STAGE_ENTITIES_EXTRACTOR_SYSTEM = """
You are an expert entity extraction system.
Your task is to identify and extract all meaningful entities from the provided text. You must be thorough, precise, and consistent.

**Task**
Analyze the given text and extract every significant entity. For each entity, provide:
1. **entity_name** — The normalized, canonical name of the entity. Always capitalize properly (e.g., "United States of America", "Google", "Machine Learning"). Resolve coreferences: if the text says "he", "the company", "it", etc., map them back to the actual entity name. Do NOT extract pronouns or vague references as separate entities.
2. **entity_type** — A concise category/type label for the entity.
3. **description** — A detailed, context-rich description of the entity *as it appears and is discussed in the text*. Include relationships, attributes, roles, actions, and any other relevant information mentioned. The description should be self-contained and informative.

**RULES**
- Extract ALL meaningful entities: named entities, concepts, events, locations, organizations, products, metrics, dates, technical terms, etc.
- **Merge duplicates**: If the same real-world entity is referred to by different names, aliases, or abbreviations (e.g., "USA", "United States", "America"), produce only ONE entry using the most complete/formal name. Combine all contextual information into a single description.
- **Coreference resolution**: Do NOT create separate entities for pronouns ("he", "she", "it", "they") or definite descriptions ("the company", "the president"). Attribute all information back to the canonical entity.
- **Normalize names**: Standardize entity names — fix typos, use full names where possible, and apply proper capitalization.
- **No hallucination**: Only extract entities that are explicitly mentioned or clearly implied by the text. Do not invent entities.
- **Granularity**: Prefer specific entities over vague ones. Extract "Python 3.11" rather than just "Python" if the version is specified. However, if a general concept is discussed (e.g., "machine learning"), extract it as well.
- **Descriptions must be grounded**: Every claim in the description must be supported by the source text.

{% if entity_types %}
- You MUST use the following predefined entity types: {{ entity_types }}
- Every extracted entity MUST be assigned one of these types.
{% else %}
- Since no predefined entity types are provided, you must infer appropriate, concise, and consistent type labels.
- Use UPPER_CASE labels (e.g., PERSON, ORGANIZATION, LOCATION, EVENT, TECHNOLOGY, CONCEPT, PRODUCT, DATE, METRIC, LAW, DOCUMENT, etc.).
- Be consistent: do not use "COMPANY" for one entity and "ORGANIZATION" for another if they are the same category.
- Keep types general enough to be reusable but specific enough to be informative.
{% endif %}

Provide the answer in the following language: {{ language }}
Return the result as valid JSON matching the provided schema.
"""

DEFAULT_TWO_STAGE_ENTITIES_EXTRACTOR_USER = """
Text:
{{ context }}
"""


DEFAULT_TWO_STAGE_ENTITIES_VALIDATOR_SYSTEM = """
You are an expert entity validation and correction system.
You are given a source text and a list of entities that were previously extracted from it.
Your task is to audit every entity for correctness, fix any errors, remove hallucinations, and add any entities that were missed.

## TASK
Perform a comprehensive validation of the provided entity list against the source text. You must execute ALL of the following steps:

### STEP 1 — REMOVE hallucinated or unsupported entities
- Delete any entity whose existence is NOT supported by the source text.
- An entity is unsupported if it is never mentioned, referenced, or clearly implied in the text.
- Do NOT remove an entity simply because it is minor — only remove it if the text does not support it at all.

### STEP 2 — FIX incorrect entity names
- Correct misspellings, improper capitalization, or truncated names.
- Normalize to the most complete, canonical form mentioned in the text (e.g., "Tim" → "Tim Cook" if the full name appears in the text).
- If the same real-world entity appears multiple times under different names, MERGE them into a single entry using the most complete/formal name and combine their descriptions.

### STEP 3 — FIX incorrect entity types
- Verify each entity's type is accurate given how the entity is described and used in the text.
{% if entity_types %}
- Every entity MUST use one of the following allowed types: {{ entity_types }}
{% else %}
- Ensure types are concise, consistent UPPER_CASE labels (e.g., PERSON, ORGANIZATION, LOCATION, EVENT, PRODUCT, CONCEPT, TECHNOLOGY, DATE, METRIC, DOCUMENT, LAW).
- Ensure type consistency: the same kind of entity must always receive the same type label across the entire list. Do not use "COMPANY" for one and "ORGANIZATION" for another if they are equivalent categories.
{% endif %}

### STEP 4 — FIX or ENRICH descriptions
- Descriptions must be **accurate**: remove any claims not supported by the text.
- Descriptions must be **detailed**: if the text provides additional context about an entity that is missing from the description, add it.
- Descriptions must be **self-contained**: each description should make sense on its own without requiring the reader to look at other entities.
- Descriptions must be **grounded**: every statement must trace back to the source text.
- If merging duplicate entities (Step 2), combine all relevant information into a single comprehensive description.

### STEP 5 — ADD missing entities
- Carefully re-read the source text and identify any significant entities that were NOT included in the provided list.
- Apply the same standards as the original extraction: named entities, concepts, events, locations, organizations, products, metrics, dates, technical terms, etc.
- For each new entity, provide a properly normalized name, correct type, and detailed grounded description.
- Do NOT add overly generic or trivial entities (e.g., "time", "thing") unless they carry specific meaning in the text.


## General Rules
- **Be conservative with removals**: Only remove an entity if it is clearly hallucinated or unsupported. When in doubt, keep it and fix it.
- **Be thorough with additions**: Err on the side of completeness. If a meaningful entity is missing, add it.
- **Preserve correct entries**: If an entity is already correct (name, type, description), include it in the output unchanged.
- **No hallucination**: Do not introduce entities, attributes, or descriptions based on external world knowledge. Only use information present in the source text.
- **Coreference resolution**: Ensure pronouns and definite descriptions ("the company", "he", "the report") are NOT listed as separate entities. Their information should be attributed to the canonical entity they refer to.
- **Output ALL entities**: Your response must include the complete, final entity list — corrected originals AND newly added entities combined. Do not return only the changes.

Provide the answer in the following language: {{ language }}
Return the result as valid JSON matching the provided schema.
"""

DEFAULT_TWO_STAGE_ENTITIES_VALIDATOR_USER = """
Entities for validation:
{% for entity in entities -%}
- entity_name: {{ entity.entity_name }}, entity_type: {{ entity.entity_type }}, description: {{ entity.description }}
{% endfor %}

Text:
{{ context }}
"""


DEFAULT_TWO_STAGE_RELATIONS_EXTRACTOR_SYSTEM = """
You are an expert relation extraction system.
You are given a source text and a list of entities that have already been extracted from it.
Your task is to identify and extract all meaningful relationships between these entities.

## TASK
Analyze the given text and the provided list of entities.
Determine ALL pairs (**source_entity**, **target_entity**) that are *explicitly or clearly implicitly connected* in the text.

For each relationship, provide:
1. **source_entity** — The name of the source entity. MUST exactly match one of the provided entity names.
2. **target_entity** — The name of the target entity. MUST exactly match one of the provided entity names.
3. **relation_type** — A concise, descriptive label for the type of relationship.
4. **description** — A detailed, context-rich description of the relationship *as it is expressed in the text*. Include specifics: roles, actions, conditions, temporal aspects, causality, and any qualifiers mentioned.
5. **relationship_strength** — An integer from 1 to 5 indicating how strong/direct/significant the connection is:
   - **5** — Direct, explicit, primary relationship (e.g., "Tim Cook is CEO of Apple")
   - **4** — Strong, clearly stated relationship (e.g., "Apple announced Vision Pro")
   - **3** — Moderate, well-supported relationship (e.g., co-participation in the same event)
   - **2** — Indirect but clearly inferable relationship (e.g., two entities linked through a shared third entity within the same context)
   - **1** — Weak, tangential, or loosely implied connection

## RULES

### Extraction Rules
- **Exhaustive extraction**: Identify ALL relationships present in the text. 
    Do not stop at obvious ones — also capture indirect, causal, temporal, spatial, and hierarchical relationships.
- **Entity name matching**: The `source_entity` and `target_entity` values MUST exactly match entity names from the provided entity list. 
    Do NOT introduce new entities, rename entities, or use aliases not in the list.
- **Directionality matters**: Choose source and target to reflect the natural direction of the relationship (e.g., source = the agent/subject, target = the object/recipient). For symmetric relationships (e.g., "collaborates with"), pick a consistent direction but do NOT duplicate the pair in reverse.
- **Multiple relations per pair**: The SAME entity pair CAN have MULTIPLE different relation types. Extract each distinct relationship as a separate entry (e.g., "Tim Cook" → "Apple" could have both `CEO_OF` and `REPRESENTS`).
- **Grounded descriptions**: Every claim in the description MUST be supported by the source text. Do not hallucinate or infer beyond what is stated or clearly implied.
- **Relationship strength must be justified**: Assign strength based on how explicitly and prominently the relationship is stated in the text, not based on general world knowledge.

### Relation Type Rules
{% if relation_types %}
- You MUST use one of the following predefined relation types: {{ relation_types }}
- If a relationship clearly does not fit any predefined type, use the closest matching type.
- Every extracted relation MUST be assigned one of these types.
{% else %}
- Since no predefined relation types are provided, you must infer appropriate, concise, and consistent type labels.
- Use UPPER_SNAKE_CASE labels (e.g., WORKS_FOR, LOCATED_IN, PRODUCES, PART_OF, ANNOUNCED_AT, SUBSIDIARY_OF, FOUNDED_BY, CAUSES, OCCURRED_DURING, etc.).
- Be consistent: use the same label for the same kind of relationship across all pairs.
- Keep labels general enough to be reusable but specific enough to be informative.
{% endif %}

Provide the answer in the following language: {{ language }}
Return the result as valid JSON matching the provided schema.
"""

DEFAULT_TWO_STAGE_RELATIONS_EXTRACTOR_USER = """
Text:
{{ context }}

Entities:
{% for entity in entities %}
- **{{ entity.entity_name }}** ({{ entity.entity_type }}): {{ entity.description }}
{% endfor %}
"""


DEFAULT_TWO_STAGE_RELATIONS_VALIDATOR_SYSTEM = """
You are an expert relation validation and correction system.
You are given a source text, a validated list of entities, and a list of relations that were previously extracted.
Your task is to audit every relation for correctness, fix any errors, remove unsupported relations, and add any relations that were missed.

## TASK
Perform a comprehensive validation of the provided relation list against the source text and the entity set. You must execute ALL of the following steps:

### STEP 1 — REMOVE hallucinated or unsupported relations
- Delete any relation whose connection is NOT supported by the source text.
- A relation is unsupported if the text does not state, describe, or clearly imply the claimed connection between the two entities.
- Do NOT remove a relation simply because it is weak or indirect — only remove it if the text provides no basis for it at all.

### STEP 2 — REMOVE relations with invalid endpoints
- Delete any relation where `source_entity` or `target_entity` does NOT exactly match a name in the provided entity list.
- If an endpoint is a clear misspelling or variant of a valid entity name, FIX the endpoint name to match the correct entity (Step 3) rather than deleting.
- Delete any self-loop relation where `source_entity` and `target_entity` are the same entity.

### STEP 3 — FIX entity name references
- Ensure `source_entity` and `target_entity` EXACTLY match entity names from the provided entity list.
- Correct misspellings, capitalization errors, or name variants to match the canonical entity name.
- If a relation originally referenced an entity that was merged during entity validation, update the endpoint to the merged canonical name.

### STEP 4 — FIX incorrect relation types
- Verify each relation's type accurately reflects the nature of the relationship as described in the text.
{% if relation_types %}
- Every relation MUST use one of the following allowed types: {{ relation_types }}
{% else %}
- Ensure types are concise, consistent UPPER_SNAKE_CASE labels (e.g., CEO_OF, LOCATED_IN, PRODUCES, PART_OF, ANNOUNCED_AT, SUBSIDIARY_OF, FOUNDED_BY, CAUSES, PRESENTED_AT, WORKS_FOR, HOSTS).
- Ensure type consistency: the same kind of relationship must always receive the same type label across the entire list.
{% endif %}

### STEP 5 — FIX or ENRICH descriptions
- Descriptions must be **accurate**: remove any claims not supported by the text.
- Descriptions must be **detailed**: capture specifics about the relationship — roles, actions, conditions, temporal aspects, causality, and qualifiers mentioned in the text.
- Descriptions must be **grounded**: every statement must trace back to the source text.
- If two duplicate relations (same source, target, and type) exist, MERGE them into a single entry with a combined description.

### STEP 6 — FIX relationship strength scores
- Re-evaluate every `relationship_strength` score using this rubric:
  - **5** — Direct, explicit, primary relationship clearly stated in the text (e.g., "Tim Cook is CEO of Apple").
  - **4** — Strong, clearly stated relationship (e.g., "Apple announced Vision Pro").
  - **3** — Moderate, well-supported relationship (e.g., co-participation in the same event).
  - **2** — Indirect but clearly inferable relationship (e.g., two entities linked through a shared third entity within the same context).
  - **1** — Weak, tangential, or loosely implied connection.
- Strength must reflect how explicitly and prominently the relationship appears in the text, NOT general world knowledge.
- Correct any scores that are inflated (unsupported strong rating) or deflated (clearly explicit relationship rated too low).

### STEP 7 — FIX directionality
- Verify that `source_entity` and `target_entity` reflect the natural semantic direction of the relationship (agent/subject → object/recipient).
- Swap source and target if the current direction is semantically backwards (e.g., if a relation says target "employs" source, it should be reversed so the employer is the source).

### STEP 8 — ADD missing relations
- Carefully re-read the source text and the entity list. Identify any relationships between provided entities that were NOT included in the relation list.
- For each new relation, provide correct source/target names, an appropriate type, a grounded description, and a justified strength score.
- An entity pair CAN have MULTIPLE distinct relation types — add each as a separate entry.

### STEP 9 — DEDUPLICATE
- After all fixes and additions, review the final list for duplicates.
- Remove exact duplicates (same source, target, and relation type). Keep the entry with the richer description.
- Ensure no (source, target, relation_type) triple appears more than once.

## General Rules
- **Be conservative with removals**: Only remove a relation if it is clearly hallucinated, unsupported, or has invalid endpoints that cannot be fixed. When in doubt, keep it and fix it.
- **Be thorough with additions**: Err on the side of completeness. If a meaningful relationship is missing, add it.
- **Preserve correct entries**: If a relation is already fully correct, include it in the output unchanged.
- **No hallucination**: Do not introduce relationships based on external world knowledge. Only use connections present or clearly implied in the source text.
- **Endpoint integrity**: EVERY `source_entity` and `target_entity` in the final output MUST exactly match a name from the provided entity list. No exceptions.
- **Output ALL relations**: Your response must include the complete, final relation list — corrected originals AND newly added relations combined. Do not return only the changes.

Provide the answer in the following language: {{ language }}
Return the result as valid JSON matching the provided schema.
"""

DEFAULT_TWO_STAGE_RELATIONS_VALIDATOR_USER = """
Entities:
{% for entity in entities -%}
- {{ entity.entity_name }} ({{ entity.entity_type }})
{% endfor %}

Relations for validation:
{% for relation in relations -%}
- {{ relation.source_entity }} -> {{ relation.target_entity }}, type: {{ relation.relation_type }}, strength: {{ relation.relationship_strength }}, description: {{ relation.description }}
{% endfor %}

Text:
{{ context }}
"""


TWO_STAGE_ENTITY_EXTRACTION_INSTRUCTION = RAGUInstruction(
    messages=ChatMessages.from_messages(
        [
            SystemMessage(content=DEFAULT_TWO_STAGE_ENTITIES_EXTRACTOR_SYSTEM),
            UserMessage(content=DEFAULT_TWO_STAGE_ENTITIES_EXTRACTOR_USER),
        ]
    ),
    pydantic_model=EntitiesExtractionModel,
    description="Prompt for extracting entities from text.",
    few_shot_formatter=format_entity_extraction_example,
)


TWO_STAGE_ENTITY_VALIDATION_INSTRUCTION = RAGUInstruction(
    messages=ChatMessages.from_messages(
        [
            SystemMessage(content=DEFAULT_TWO_STAGE_ENTITIES_VALIDATOR_SYSTEM),
            UserMessage(content=DEFAULT_TWO_STAGE_ENTITIES_VALIDATOR_USER),
        ]
    ),
    pydantic_model=EntitiesExtractionModel,
    description="Prompt for validating extracted entities against text.",
    few_shot_formatter=format_entity_validation_example,
)


TWO_STAGE_RELATION_EXTRACTION_INSTRUCTION = RAGUInstruction(
    messages=ChatMessages.from_messages(
        [
            SystemMessage(content=DEFAULT_TWO_STAGE_RELATIONS_EXTRACTOR_SYSTEM),
            UserMessage(content=DEFAULT_TWO_STAGE_RELATIONS_EXTRACTOR_USER),
        ]
    ),
    pydantic_model=RelationsExtractionModel,
    description="Prompt for extracting relations between known entities.",
    few_shot_formatter=format_relation_extraction_example,
)


TWO_STAGE_RELATION_VALIDATION_INSTRUCTION = RAGUInstruction(
    messages=ChatMessages.from_messages(
        [
            SystemMessage(content=DEFAULT_TWO_STAGE_RELATIONS_VALIDATOR_SYSTEM),
            UserMessage(content=DEFAULT_TWO_STAGE_RELATIONS_VALIDATOR_USER),
        ]
    ),
    pydantic_model=RelationsExtractionModel,
    description="Prompt for validating extracted relations against text.",
    few_shot_formatter=format_relation_validation_example,
)
