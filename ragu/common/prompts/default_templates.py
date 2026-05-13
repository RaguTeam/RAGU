DEFAULT_ARTIFACTS_EXTRACTOR_SYSTEM = """
**-Goal-**
A text document and a list of entity types are given.
It is necessary to identify all entities of the specified types in the text, as well as all relationships between the identified entities.

**-Steps-**
1. **Identify all entities.**
    For each detected entity, extract the following information:
    - **entity_name**: The normalized name of the entity, starting with a capital letter.
        Normalization means reducing the word to its base form.
        Example: рождеству → Рождество, кошек → Кошки, Павла → Павел.
    - **entity_type**: The type of the entity.
    {% if entity_types -%}
        The entity type must be one of the following: {{ entity_types }}
    {% endif %}
    - **description**: A detailed description of the entity according to the given text. The description must be precise and as complete as possible.

2. **Determine relationships between entities.**
    Based on the entities identified in step 1, determine all pairs (**source_entity**, **target_entity**) that are *explicitly connected* to each other.
    For each such pair, extract the following information:
    - **source_entity**: The name of the source entity (as defined in step 1).
    - **target_entity**: The name of the target entity (as defined in step 1).
    - **relation_type**: The type of the relation.
    {% if relation_types -%}
        The relation type must be one of the following: {{ relation_types }}
    {% endif %}
    - **description**: A description of the relationship between the two entities.
    - **relationship_strength**: A numeric value representing the strength of the relationship between the entities, ranging from 0 to 5,
    where 0 = weak connection and 5 = strong connection.

Provide the answer in the following language: {{ language }}
Return the result as valid JSON matching the provided schema.
"""

DEFAULT_ARTIFACTS_EXTRACTOR_USER = """
Text:
{{ context }}
"""

DEFAULT_ARTIFACTS_VALIDATOR_SYSTEM = """
**Goal**
Validate correctness and completeness of entities and relationships against the given text.

**Instructions**
1. Add missing entities with correct types and descriptions.
2. Add missing relationships with descriptions and strength.
3. Return full updated lists.

{% if entity_types -%}
The entity type must be one of the following: {{ entity_types }}
{% endif %}

Provide the answer in the following language: {{ language }}
Return the result as valid JSON matching the provided schema.
"""

DEFAULT_ARTIFACTS_VALIDATOR_USER = """
Triplets for validation:
{{ artifacts }}

Text for validation:
{{ context }}
"""

DEFAULT_COMMUNITY_REPORT_PROMPT = """
**Goal**
Generate a detailed community report using entities, their relationships, and any additional statements.

**Instructions**
1. Create a clear title and summary.
2. Provide an impact rating with justification.
3. Produce 5–10 key findings with short summaries and detailed explanations.

Input text:
{% for entity in community.entities -%}
Entity: {{ entity.entity_name }}, description: {{ entity.description }}{% if not loop.last %}, {% endif %}
{% endfor %}

Relations
{% for relation in community.relations -%}
{{ relation.subject_name }} -> {{ relation.object_name }}, relations description: {{ relation.description }}
{% endfor %}

Provide the answer in the following language: {{ language }}
Return the result as valid JSON matching the provided schema.
"""

DEFAULT_RELATIONSHIP_SUMMARIZER_PROMPT = """
**Goal**
From the given entity pair and multiple phrases, produce one concise, consistent relationship description.

Data:
Subject: {{ relation.subject_name }}, Object: {{ relation.object_name }}, Relationship description: {{ relation.description }}

Provide the answer in the following language: {{ language }}
Return the result as valid JSON matching the provided schema.
"""

# "Entity" -> "Entity name" to better match schema, and reduce json validation errors
DEFAULT_ENTITY_SUMMARIZER_PROMPT = """
**Goal**
From the given entity and multiple phrases, produce one concise, consistent entity description.

Data:
Entity name: {{ entity.entity_name }}, Description: {{ entity.description }}

Provide the answer in the following language: {{ language }}
Return the result as valid JSON matching the provided schema.
"""

DEFAULT_GLOBAL_SEARCH_CONTEXT_PROMPT = """
**Goal**
Answer the query by summarizing relevant information from the context and, if needed, well-known facts.

**Instructions**
1. Reason about context relevance.
2. Provide a usefulness rating from 0 to 10 (0 = useless, 10 = direct answer).

Query: {{ query }}
Context: {{ context }}

Provide the answer in the following language: {{ language }}
Return the result as valid JSON matching the provided schema.
"""

DEFAULT_GLOBAL_SEARCH_PROMPT = """
**Goal**
Answer the query by summarizing the provided ranked context.

**Instructions**
1. Consider the relevance ranking (lower rank = less relevant).
2. Briefly reason about context relevance before giving the final answer.

Query: {{ query }}
Context: {{ context }}

Provide the answer in the following language: {{ language }}
"""

DEFAULT_RESPONSE_ONLY_PROMPT = """
**Goal**
Answer the query by summarizing relevant information from the context and, if necessary, well-known facts.

**Instructions**
1. If you do not know the correct answer, explicitly state that.
2. Do not include unsupported information.

Query: {{ query }}
Context: {{ context }}

Provide the answer in the following language: {{ language }}
"""

DEFAULT_MIX_SEARCH_PROMPT = """
**Goal**
Answer the query by combining information gathered from multiple search engines.

**Instructions**
1. Compare the {{ section_label }}s from different engines before answering.
2. Prefer information that is supported by multiple engines.
3. If the engines conflict, mention the conflict briefly and avoid unsupported claims.
4. If you do not know the correct answer, explicitly state that.
5. Do not include unsupported information.
6. Treat the provided material as synthesized engine output; do not assume every engine used the same retrieval strategy.

Query: {{ query }}
Engine {{ section_label }}s:
{{ context }}

Provide the answer in the following language: {{ language }}
"""

DEFAULT_MIX_SEARCH_CONTEXT_PROMPT = """
{%- for result in payload.entries %}
{%- if result is not none %}
**Engine {{ loop.index }} {{ section_label }}**
{{ result }}
{% endif %}
{%- endfor %}
"""

DEFAULT_CLUSTER_SUMMARIZER_PROMPT = """
**Goal**
You are given a list of descriptions.  
Summarize them into a single concise description.  

Texts to summarize:  
{{ content }}
"""

DEFAULT_RAGU_LM_SYSTEM_PROMPT = "Вы - эксперт в области анализа текстов и извлечения семантической информации из них."

DEFAULT_RAGU_LM_ENTITY_EXTRACTION_PROMPT = """
Распознайте все именованные сущности в тексте и выпишите их список с новой строки.
Текст: {{ text }}
Именованные сущности:
"""

DEFAULT_RAGU_LM_ENTITY_NORMALIZATION_PROMPT = """
Выполните нормализацию именованной сущности, встретившейся в тексте.
Исходная (ненормализованная) именованная сущность: {{ source_entity }}
Текст: {{ source_text }}
Нормализованная именованная сущность:
"""

DEFAULT_RAGU_LM_ENTITY_DESCRIPTION_PROMPT = """
Напишите, что означает именованная сущность в тексте, то есть раскройте её смысл относительно текста.
Именованная сущность: {{ normalized_entity }}
Текст: {{ source_text }}
Смысл именованной сущности:
"""

DEFAULT_RAGU_LM_RELATION_DESCRIPTION_PROMPT = """
Напишите, что означает отношение между двумя именованными сущностями в тексте, то есть раскройте смысл этого отношения относительно текста (либо напишите прочерк, если между двумя именованными сущностями отсутствует отношение).
Первая именованная сущность: {{ first_normalized_entity }}
Вторая именованная сущность: {{ second_normalized_entity }}
Текст: {{ source_text }}
Смысл отношения между двумя именованными сущностями:
"""

DEFAULT_QUERY_DECOMPOSITION_PROMPT = """
**Goal**
You are a Query Planning agent for a Retrieval-Augmented Generation (RAG) system.

Your task is to analyze a user's natural-language query and convert it into a structured query plan.

**Instructions**
1. Decompose the original query into a set of minimal, atomic subqueries.
   - Each subquery should represent a single information need.
   - Subqueries should be as independent as possible.
2. Identify dependencies between subqueries.
   - If a subquery requires the result of another subquery, explicitly specify this dependency.
   - Dependencies must form a directed acyclic graph (DAG).
3. Assign each subquery a unique identifier.
4. Optionally classify each subquery by its intent (e.g., factual lookup, comparison, aggregation, reasoning).

**Rules**
- Do NOT answer the query.
- Do NOT invent information.
- Do NOT merge unrelated information needs into one subquery.
- If query does not consist of subqueries, return original query as only subquery.
- Dependencies should be explicit and minimal.

Query to decompose: {{ query }}

Output only valid JSON that strictly conforms to the provided schema.
"""

DEFAULT_QUERY_REWRITE_PROMPT = """
**Goal**
You are a query rewriting assistant for a Retrieval-Augmented Generation (RAG) system.

Your task is to rewrite a subquery so that it becomes fully explicit and self-contained,
using the answers to its dependency subqueries.

**Rules**
- Preserve the original intent of the subquery.
- Resolve all references (pronouns, placeholders, implicit entities).
- Do NOT add new information.
- Do NOT answer the question.
- Output only the rewritten query as plain text.

Original subquery:
{{ original_query }}

Dependency answers:
{% for dep_id in context -%}
{{ dep_id }}: {{ context[dep_id] }}
{% endfor %}
Rewrite the subquery and return the result as valid JSON matching the provided schema.
"""
