# RAGU

---

1. [Terminology](#terminology)
2. [Introduction to the Knowledge Graph Construction Methodology](#introduction-to-the-knowledge-graph-construction-methodology)

   1. [Text Chunking](#1-text-chunking)
   2. [Entity, Relation, and Description Extraction](#2-entity-relation-and-description-extraction)
   3. [Triplet Processing](#3-triplet-processing)
   4. [Graph Construction and Community Detection](#4-graph-construction-and-community-detection)
3. [RAG](#rag)

   1. [Local Search](#local-search)
   2. [Global Search](#global-search)
4. [RAGU Components](#ragu-components)

   1. [Chunkers](#chunkers)
   2. [Graph Extraction Pipeline](#graph-extraction-pipeline)
   3. [Knowledge Graph](#knowledge-graph)
   4. [Knowledge Graph Search](#knowledge-graph-search)

---

## Terminology

1. **Chunk** — a fragment of text obtained by splitting the original corpus into smaller parts.
2. **Chunking** — the process of dividing text into fragments (chunks).
3. **Entity (node/vertex)** — an object extracted from text that has independent meaning (e.g., a person, organization, event, or location). In a graph, it is represented as a vertex.
4. **Relation (edge/link)** — a semantic connection between two entities (e.g., *“Peter I — ruler → Russian Empire”*). In a graph, it is represented as an edge.
5. **Community** — a subgroup of vertices that are densely interconnected internally but weakly connected to the rest of the graph (cluster/community).
6. **Summarization** — the process of generating a concise, unified representation of a text or a set of descriptions.
7. **Abstractive question** — a question that requires aggregating and connecting information from across the entire corpus (e.g., “What was each podcast speaker’s opinion on using AI in medicine?”).
8. **Extractive question** — a question that can be answered by locating a specific fragment in the text (e.g., “In what year was Peter I born?”).

---

## Introduction to the Knowledge Graph Construction Methodology

### 1. Text Chunking

To make processing more efficient, the input text corpus is divided into small fragments.
RAGU supports several chunking strategies:

* **`SimpleChunker`** — fixed-length splitting by text size.
* **`SemanticTextChunker`** — chunking that preserves semantic coherence between sentences.
* **`SmartSemanticChunker`** — an advanced semantic chunking approach based on `smart_chunker`.

---

### 2. Entity, Relation, and Description Extraction

For each text fragment, structured information is extracted:

* **Entities** — textual representation, entity type, and contextual description.
* **Relations** — description of the semantic link between two entities (or a relation class), and its confidence score.

> **RAGU uses entity and relation classes from [NEREL](https://github.com/nerel-ds/NEREL).**
>
> `Entity` and `Relation` are base graph model classes. They can be inherited for domain-specific nodes and edges as long as storage adapters can still operate on the `Node` / `Edge` base contracts.

**Entity classes:**

| No. | Entity type | No. | Entity type  | No. | Entity type   |
| --- | ----------- | --- | ------------ | --- | ------------- |
| 1.  | AGE         | 11. | FAMILY       | 21. | PENALTY       |
| 2.  | AWARD       | 12. | IDEOLOGY     | 22. | PERCENT       |
| 3.  | CITY        | 13. | LANGUAGE     | 23. | PERSON        |
| 4.  | COUNTRY     | 14. | LAW          | 24. | PRODUCT       |
| 5.  | CRIME       | 15. | LOCATION     | 25. | PROFESSION    |
| 6.  | DATE        | 16. | MONEY        | 26. | RELIGION      |
| 7.  | DISEASE     | 17. | NATIONALITY  | 27. | STATE_OR_PROV |
| 8.  | DISTRICT    | 18. | NUMBER       | 28. | TIME          |
| 9.  | EVENT       | 19. | ORDINAL      | 29. | WORK_OF_ART   |
| 10. | FACILITY    | 20. | ORGANIZATION |     |               |

**Relation classes:**

| No. | Relation type    | No. | Relation type      | No. | Relation type    |
| --- | ---------------- | --- | ------------------ | --- | ---------------- |
| 1.  | ABBREVIATION     | 18. | HEADQUARTERED_IN   | 35. | PLACE_RESIDES_IN |
| 2.  | AGE_DIED_AT      | 19. | IDEOLOGY_OF        | 36. | POINT_IN_TIME    |
| 3.  | AGE_IS           | 20. | INANIMATE_INVOLVED | 37. | PRICE_OF         |
| 4.  | AGENT            | 21. | INCOME             | 38. | PRODUCES         |
| 5.  | ALTERNATIVE_NAME | 22. | KNOWS              | 39. | RELATIVE         |
| 6.  | AWARDED_WITH     | 23. | LOCATED_IN         | 40. | RELIGION_OF      |
| 7.  | CAUSE_OF_DEATH   | 24. | MEDICAL_CONDITION  | 41. | SCHOOLS_ATTENDED |
| 8.  | CONVICTED_OF     | 25. | MEMBER_OF          | 42. | SIBLING          |
| 9.  | DATE_DEFUNCT_IN  | 26. | ORGANIZES          | 43. | SPOUSE           |
| 10. | DATE_FOUNDED_IN  | 27. | ORIGINS_FROM       | 44. | START_TIME       |
| 11. | DATE_OF_BIRTH    | 28. | OWNER_OF           | 45. | SUBEVENT_OF      |
| 12. | DATE_OF_CREATION | 29. | PARENT_OF          | 46. | SUBORDINATE_OF   |
| 13. | DATE_OF_DEATH    | 30. | PART_OF            | 47. | TAKES_PLACE_IN   |
| 14. | END_TIME         | 31. | PARTICIPANT_IN     | 48. | WORKPLACE        |
| 15. | EXPENDITURE      | 32. | PENALIZED_AS       | 49. | WORKS_AS         |
| 16. | FOUNDED_BY       | 33. | PLACE_OF_BIRTH     |     |                  |
| 17. | HAS_CAUSE        | 34. | PLACE_OF_DEATH     |     |                  |

In the standard extraction pipeline, NEREL entity and relation types are injected into extraction prompts by default. You can pass custom `entity_types` and `relation_types` to the LLM extractors when you need a different ontology.

---

### 3. `Triplet` Processing

Due to text diversity and parallel extraction, duplicate entities and relations may appear.
To unify them, RAGU performs aggregation and summarization:

1. All descriptions of the same entity or relation are aggregated.
2. Optionally, aggregated texts are summarized by an LLM to produce a single, consistent description.

**Example:**

* Input:
 *Пётр Первый* → "русский царь"
 * *Пётр Великий* → "реформатор"
* Summarized:
  * *Пётр I (Пётр Великий)* → "Император России, проводивший масштабные реформы"

Responsible components:

* **`EntitySummarizer`** — aggregation and summarization of entity descriptions.
* **`RelationSummarizer`** — aggregation and summarization of relations.

> **TODO:** ontology-based entity alignment that merges different surface forms of the same entity (e.g., *“Peter I”*, *“Emperor Peter the Great”*).
> See [Chepurova A. et al., 2024](https://aclanthology.org/2024.textgraphs-1.5.pdf).

---

### 4. Graph Construction and Community Detection

After entity and relation extraction, all elements are merged into a unified graph.
The persisted graph is a directed multigraph: relations preserve source entity, target entity, and relation identity. Community detection builds an undirected projection internally for clustering.

To enable **abstractive question answering**, RAGU follows the **GraphRAG methodology**, where the graph is clustered into **communities** and summarized.
Community detection is performed using the [Leiden algorithm](https://en.wikipedia.org/wiki/Leiden_algorithm).

Each community summary has the following structure:

```json
{
    "title": "<report_title>",
    "summary": "<brief_description>",
    "rating": "<importance_score>",
    "rating_explanation": "<rationale_for_rating>",
    "findings": [
        {
            "summary": "<short finding description>",
            "explanation": "<detailed explanation>"
        },
        {
            "summary": "<short finding description>",
            "explanation": "<detailed explanation>"
        }
    ]
}
```

Community summarization is handled by the instruction `community_report`.
During query processing, these summaries are transformed into textual context for the LLM.

---

## RAG

### Local Search

Local search is best suited for **fine-grained, extractive queries**—the “needle-in-a-haystack” type of problem (e.g., “When did Alexander II die?”).

Workflow:

1. Retrieve relevant entities for the query.
2. Collect all related relations and neighboring entities.
3. Collect relevant community summaries.
4. Retrieve original chunks where those entities appeared.

Entity relevance is determined by the cosine similarity between query embeddings and entity embeddings.

**Context structure example:**

```markdown
**Сущности**
Сущность, тип сущности, описание сущности
...

**Отношения**
Сущность-источник, целевая сущность, описание отношения, ранг отношения
...

**Саммари**
...

**Тексты**
...
```

The query and the gathered context are then passed to the LLM to produce the final answer.
Handled by the instruction `local_search`.

---

### Global Search

Global search operates at the **community level**:

1. Get all communities.
2. Generate a **meta-answer** for each community, each with a relevance rating between 0 and 10.
3. Sort meta-answers by relevance and use them as higher-level context for the LLM to produce a global response.

**Meta-answer structure:**

```json
{
    "reasoning": "<explanation of context relevance>",
    "response": "<generated answer>",
    "rating": "<relevance score>"
}
```

Meta-answer generation is controlled by `global_search_context`, and final synthesis by `global_search`.
Note: Global search is a **computationally expensive** operation.

---

## RAGU Components

### Chunkers

The first step in graph construction is splitting raw text corpora into chunks.
RAGU implements several chunking strategies and supports user-defined ones.

The chunker interface is defined in `BaseChunker` (`ragu/chunker/base_chunker.py`).

RAGU has a naive chunker and semantic chunkers:

* **`SimpleChunker`** — fixed-size text splitting.
* **`SemanticTextChunker`** — sentence-aware semantic splitting.
* **`SmartSemanticChunker`** — semantic splitting backed by `smart_chunker`.

The advanced semantic chunker is implemented as `SmartSemanticChunker`.
See details at: [smart_chunker GitHub](https://github.com/bond005/smart_chunker/tree/main).

---

### Graph Extraction Pipeline

RAGU supports three extraction strategies:

1. Single-step LLM extraction with `ArtifactsExtractorLLM`.
2. Two-stage LLM extraction with `TwoStageArtifactsExtractorLLM`.
3. a **lightweight fine-tuned model approach** with [RAGU-lm](https://huggingface.co/RaguTeam/RAGU-lm) (only for Russian language)

#### ArtifactsExtractorLLM

`ArtifactsExtractorLLM` implements the standard GraphRAG method for entity and relation extraction via LLMs.

```python
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI
from ragu.triplet import ArtifactsExtractorLLM, TwoStageArtifactsExtractorLLM

client = CachedAsyncOpenAI(
    base_url="https://api.openai.com/v1",
    api_key="dummy-api-token",
    rate_max_simultaneous=10,
    rate_max_per_minute=100,
)
llm = LLMOpenAI(
    client=client,
    model_name="gpt-4o-mini",
)

single_step_extractor = ArtifactsExtractorLLM(
    llm=llm,
    do_validation=True,
)

two_stage_extractor = TwoStageArtifactsExtractorLLM(
    llm=llm,
    do_entity_validation=True,
    do_relation_validation=True,
)
```

Single-step extraction and validation are driven by the `artifact_extraction` and `artifact_validation` prompts. Two-stage extraction uses `entity_extraction`, `entity_validation`, `relation_extraction`, and `relation_validation`.

#### RAGU-lm

This pipeline uses a fine-tuned model specialized for structured information extraction — [RAGU-lm](https://huggingface.co/RaguTeam/RAGU-lm).

The model is trained to perform the following tasks:

1. Recognize unnormalized named entities from text.
2. Normalize each unnormalized entity with respect to the given text.
3. Generate a definition for each normalized named entity based on the text.
4. Generate a definition of the directed relation between a pair of normalized named entities within the text.

**System prompt**:
```
Вы - эксперт в области анализа текстов и извлечения семантической информации из них.
```

**Entity extraction**:
```
Распознайте все именованные сущности в тексте и выпишите их список с новой строки.\n\nТекст: {input_text}\n\nИменованные сущности:
```

<details>
<summary>Example</summary>

**Input**:
Распознайте все именованные сущности в тексте и выпишите их список с новой строки.

Текст: Семья Обамы приобрела дом в Вашингтоне за 8,1 млн долларов Барак Обама Бывший президент США Барак Обама с женой Мишель приобрели жильё в Вашингтоне недалеко от Белого дома. После окончания срока работы на посту президента в январе супруги арендовали особняк в стиле эпохи Тюдоров в престижном районе столицы США — Калорама, где долгое время селились дипломаты, лоббисты и политики. Младшей дочери Барака и Мишель — Саше осталось ещё два года до окончания частной школы в Вашингтоне. Поэтому семья решила выкупить дом с восемью спальными комнатами стоимостью 8,1 миллиона долларов. Ранее дом принадлежал бывшему пресс-секретарю экс-президента США Билла Клинтона Джо Локхарту (), который в настоящее время возглавляет пресс-службу Национальной футбольной лиги США. Среди соседей Обамы — видные деятели Вашингтона, включая дочь действующего президента Иванку Трамп и её мужа Джареда Кушнера (оба являются советниками Белого дома), которые переехали в район Калорама из Нью-Йорка в начале 2017 года. Госсекретарь США Рекс Тиллерсон тоже живёт в доме неподалеку. «Учитывая, что президент и миссис Обама планируют жить в Вашингтоне как минимум ещё два года, для них имело смысл приобрести дом в собственность, а не продолжать его арендовать», — сообщил представитель Обамы. Семья президента Обамы также владеет домом в Чикаго — третьем по величине городе в США, где Обама приобрёл значительную политическую поддержку перед тем, как дважды стал главою Белого дома.

Именованные сущности: 

**Expected output**:

Семья Обамы
Обамы
приобрела дом
Вашингтоне
8,1 млн долларов
Барак Обама
Обама
президент США
президент
США
Мишель
приобрели жильё
Белого дома
президента
в январе
Тюдоров
столицы США
Калорама
дипломаты
лоббисты
политики
Барака
Саше
два года
восемью
8,1 миллиона долларов
пресс-секретарю экс-президента США
пресс-секретарю
президента США
Билла Клинтона
Джо Локхарту
пресс-службу Национальной футбольной лиги США
Национальной футбольной лиги США
Вашингтона
Иванку Трамп
Джареда Кушнера
советниками Белого дома
Нью-Йорка
в начале 2017 года
Госсекретарь США
Госсекретарь
Рекс Тиллерсон
миссис Обама
представитель Обамы
Семья президента Обамы
Чикаго
третьем
главою Белого дома
</details>


**Normalization**:
```
Выполните нормализацию именованной сущности, встретившейся в тексте.\n\nИсходная (ненормализованная) именованная сущность: {source_entity}\n\nТекст: {source_text}\n\nНормализованная именованная сущность:
```

<details>
<summary>Example</summary>

**Input**:
Выполните нормализацию именованной сущности, встретившейся в тексте.

Исходная (ненормализованная) именованная сущность: пресс-секретарю

Текст: Семья Обамы приобрела дом в Вашингтоне за 8,1 млн долларов Барак Обама Бывший президент США Барак Обама с женой Мишель приобрели жильё в Вашингтоне недалеко от Белого дома. После окончания срока работы на посту президента в январе супруги арендовали особняк в стиле эпохи Тюдоров в престижном районе столицы США — Калорама, где долгое время селились дипломаты, лоббисты и политики. Младшей дочери Барака и Мишель — Саше осталось ещё два года до окончания частной школы в Вашингтоне. Поэтому семья решила выкупить дом с восемью спальными комнатами стоимостью 8,1 миллиона долларов. Ранее дом принадлежал бывшему пресс-секретарю экс-президента США Билла Клинтона Джо Локхарту (), который в настоящее время возглавляет пресс-службу Национальной футбольной лиги США. Среди соседей Обамы — видные деятели Вашингтона, включая дочь действующего президента Иванку Трамп и её мужа Джареда Кушнера (оба являются советниками Белого дома), которые переехали в район Калорама из Нью-Йорка в начале 2017 года. Госсекретарь США Рекс Тиллерсон тоже живёт в доме неподалеку. «Учитывая, что президент и миссис Обама планируют жить в Вашингтоне как минимум ещё два года, для них имело смысл приобрести дом в собственность, а не продолжать его арендовать», — сообщил представитель Обамы. Семья президента Обамы также владеет домом в Чикаго — третьем по величине городе в США, где Обама приобрёл значительную политическую поддержку перед тем, как дважды стал главою Белого дома.

Нормализованная именованная сущность: 

**Expected output**:

пресс-секретарь
</details>

**Entity description generation**:

```
Напишите, что означает именованная сущность в тексте, то есть раскройте её смысл относительно текста.\n\nИменованная сущность: {normalized_entity}\n\nТекст: {source_text}\n\nСмысл именованной сущности:
```

<details>
<summary>Example</summary>

**Input**:
Напишите, что означает именованная сущность в тексте, то есть раскройте её смысл относительно текста.

Именованная сущность: пресс-секретарь

Текст: Семья Обамы приобрела дом в Вашингтоне за 8,1 млн долларов Барак Обама Бывший президент США Барак Обама с женой Мишель приобрели жильё в Вашингтоне недалеко от Белого дома. После окончания срока работы на посту президента в январе супруги арендовали особняк в стиле эпохи Тюдоров в престижном районе столицы США — Калорама, где долгое время селились дипломаты, лоббисты и политики. Младшей дочери Барака и Мишель — Саше осталось ещё два года до окончания частной школы в Вашингтоне. Поэтому семья решила выкупить дом с восемью спальными комнатами стоимостью 8,1 миллиона долларов. Ранее дом принадлежал бывшему пресс-секретарю экс-президента США Билла Клинтона Джо Локхарту (), который в настоящее время возглавляет пресс-службу Национальной футбольной лиги США. Среди соседей Обамы — видные деятели Вашингтона, включая дочь действующего президента Иванку Трамп и её мужа Джареда Кушнера (оба являются советниками Белого дома), которые переехали в район Калорама из Нью-Йорка в начале 2017 года. Госсекретарь США Рекс Тиллерсон тоже живёт в доме неподалеку. «Учитывая, что президент и миссис Обама планируют жить в Вашингтоне как минимум ещё два года, для них имело смысл приобрести дом в собственность, а не продолжать его арендовать», — сообщил представитель Обамы. Семья президента Обамы также владеет домом в Чикаго — третьем по величине городе в США, где Обама приобрёл значительную политическую поддержку перед тем, как дважды стал главою Белого дома.

Смысл именованной сущности: 

**Expected output**:

Бывший представитель СМИ экс-президента США Билла Клинтона.
</details>

**Relation extraction**:

```
Напишите, что означает отношение между двумя именованными сущностями в тексте, то есть раскройте смысл этого отношения относительно текста (либо напишите прочерк, если между двумя именованными сущностями отсутствует отношение).\n\nПервая именованная сущность: {first_normalized_entity}\n\nВторая именованная сущность: {second_normalized_entity}\n\nТекст: {source_text}\n\nСмысл отношения между двумя именованными сущностями:
```

<details>
<summary>Example</summary>

**Input:**
Напишите, что означает отношение между двумя именованными сущностями в тексте, то есть раскройте смысл этого отношения относительно текста (либо напишите прочерк, если между двумя именованными сущностями отсутствует отношение).

Первая именованная сущность: Джо Локхарт

Вторая именованная сущность: пресс-служба Национальной футбольной лиги США

Текст: Семья Обамы приобрела дом в Вашингтоне за 8,1 млн долларов Барак Обама Бывший президент США Барак Обама с женой Мишель приобрели жильё в Вашингтоне недалеко от Белого дома. После окончания срока работы на посту президента в январе супруги арендовали особняк в стиле эпохи Тюдоров в престижном районе столицы США — Калорама, где долгое время селились дипломаты, лоббисты и политики. Младшей дочери Барака и Мишель — Саше осталось ещё два года до окончания частной школы в Вашингтоне. Поэтому семья решила выкупить дом с восемью спальными комнатами стоимостью 8,1 миллиона долларов. Ранее дом принадлежал бывшему пресс-секретарю экс-президента США Билла Клинтона Джо Локхарту (), который в настоящее время возглавляет пресс-службу Национальной футбольной лиги США. Среди соседей Обамы — видные деятели Вашингтона, включая дочь действующего президента Иванку Трамп и её мужа Джареда Кушнера (оба являются советниками Белого дома), которые переехали в район Калорама из Нью-Йорка в начале 2017 года. Госсекретарь США Рекс Тиллерсон тоже живёт в доме неподалеку. «Учитывая, что президент и миссис Обама планируют жить в Вашингтоне как минимум ещё два года, для них имело смысл приобрести дом в собственность, а не продолжать его арендовать», — сообщил представитель Обамы. Семья президента Обамы также владеет домом в Чикаго — третьем по величине городе в США, где Обама приобрёл значительную политическую поддержку перед тем, как дважды стал главою Белого дома.

Смысл отношения между двумя именованными сущностями: 

**Expected output:**
Возглавляет пресс-службу Национальной футбольной лиги США.
</details>


**How to use?:**

It is recommended to use vllm to run ragu-lm.
https://docs.vllm.ai/en/latest/

Run vllm.
```bash
sudo vllm serve RaguTeam/ragu-lm --max_model_len 4096 
```

Initialize RaguLmArtifactExtractor and use it.
```python
from ragu.chunker.types import Chunk
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI
from ragu.triplet.ragu_lm_artifact_extractor import RaguLmArtifactExtractor

client = CachedAsyncOpenAI(
    base_url="http://0.0.0.0:8000/v1/",
    api_key="dummy-api-token",
)
llm = LLMOpenAI(
    client=client,
    model_name="RaguTeam/ragu-lm",
)

pipeline = RaguLmArtifactExtractor(
    llm=llm,
)

chunks = [Chunk(content="Some source text.", chunk_order_idx=0, doc_id="doc-1")]
entities, relations = await pipeline.extract(chunks)
```

---

### Knowledge Graph

For efficient retrieval, all graph elements are indexed and stored.
The main abstraction for managing this data is the `KnowledgeGraph` class.

```python
from ragu import BuilderArguments, KnowledgeGraph, SimpleChunker
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI
from ragu.triplet import TwoStageArtifactsExtractorLLM

# Shared client — works for small-to-medium corpora
client = CachedAsyncOpenAI(
    base_url="https://api.openai.com/v1",
    api_key="dummy-api-token",
    rate_max_simultaneous=10,
    rate_max_per_minute=100,
)

# For large corpora (thousands of entities), use separate clients:
#
# llm_client = CachedAsyncOpenAI(
#     base_url="https://api.openai.com/v1",
#     api_key="dummy-api-token",
#     rate_max_simultaneous=5,
#     rate_max_per_minute=60,
# )
# embed_client = CachedAsyncOpenAI(
#     base_url="https://api.openai.com/v1",
#     api_key="dummy-api-token",
#     rate_max_simultaneous=20,
#     rate_max_per_minute=500,
#     embed_timeout=60.0,
# )

llm = LLMOpenAI(
   client=client,
   model_name="gpt-4o-mini"
)
embedder = EmbedderOpenAI(
    client=client,
    model_name="text-embedding-3-large",
    dim=3072,
)

chunker = SimpleChunker(max_chunk_size=1000)
artifact_extractor = TwoStageArtifactsExtractorLLM(llm=llm)
builder_settings = BuilderArguments(
    use_llm_summarization=True,
    use_clustering=True,
    cluster_only_if_more_than=2,
    make_community_summary=True,
)

knowledge_graph = KnowledgeGraph(
    llm=llm,
    embedder=embedder,
    chunker=chunker,
    artifact_extractor=artifact_extractor,
    builder_settings=builder_settings,
    language="russian",
)

await knowledge_graph.build_from_docs(["Text document to index."])
```

Each entity, relation, community summary, and source chunk is stored through the configured storage adapters. Dense vectors are produced by `embedder`; optional sparse vectors are produced by `sparse_embedder` and used for hybrid search.

---

### Knowledge Graph Search

#### Local Search

Local search acts as a **granular vector-based RAG**.
It retrieves top-k relevant entities, then expands their neighborhoods:

1. All neighboring entities
2. All relations involving them
3. Their community summaries
4. The original chunks from which they were extracted

This structured context is used to answer the user’s query.

```python
from ragu.search_engine import LocalSearchEngine

local_search = LocalSearchEngine(
    llm=llm,                         # LLM for the final answer
    knowledge_graph=knowledge_graph, # Knowledge graph
    embedder=embedder,               # Embedder for query vectorization
)

search_result = await local_search.a_search("Who wrote the novel 'Quo Vadis'?")
result = await local_search.a_query("Who wrote the novel 'Quo Vadis'?")
print(result.response)
```

#### Global Search

Global search operates on community summaries:

1. Get all communities.
2. Generate a **meta-answer** for each community, each with a relevance rating between 0 and 10.
3. Sort meta-answers by relevance and use them as higher-level context for the LLM to produce a global response.


```python
from ragu.search_engine import GlobalSearchEngine

global_search = GlobalSearchEngine(
    llm=llm,
    knowledge_graph=knowledge_graph,
)

search_result = await global_search.a_search("Who wrote the novel 'Quo Vadis'?")
result = await global_search.a_query("Who wrote the novel 'Quo Vadis'?")
print(result.response)
```

#### Naive Search

Naive search is vector RAG over source chunks. It does not expand through graph neighborhoods; it retrieves relevant chunks from vector storage and uses them as context for the LLM.

```python
from ragu.search_engine import NaiveSearchEngine

naive_search = NaiveSearchEngine(
    llm=llm,
    knowledge_graph=knowledge_graph,
    embedder=embedder,
)

search_result = await naive_search.a_search("Who wrote the novel 'Quo Vadis'?")
result = await naive_search.a_query("Who wrote the novel 'Quo Vadis'?")
print(result.response)
```

#### Mix Search

Mix search runs several search engines and asks the LLM to synthesize a final answer from their responses. It is useful when you want graph neighborhood context, chunk-level vector context, and community-summary context in one answer.

```python
from ragu.search_engine import MixSearchEngine

mix_search = MixSearchEngine(
    llm=llm,
    engines=[local_search, naive_search, global_search],
)

result = await mix_search.a_query("Who wrote the novel 'Quo Vadis'?")
print(result.response)
```

#### Query Planning

`QueryPlanEngine` wraps any search engine and decomposes complex questions into dependent subqueries before producing the final answer.

```python
from ragu.search_engine import QueryPlanEngine

planned_search = QueryPlanEngine(local_search)

result = await planned_search.a_query(
    "Who wrote the novel 'Quo Vadis' and what country was the author from?"
)
print(result.response)
```

Example of a decomposed query:

```python
subqueries = await planned_search.process_query(
    "Who wrote the novel 'Quo Vadis' and what country was the author from?"
)

for subquery in subqueries:
    print(subquery.id, subquery.query, subquery.depends_on)

# Possible output:
# q1 Who wrote the novel 'Quo Vadis'? []
# q2 What country was the author from? ['q1']
```

---

### Prompt Tuning

Every LLM-powered component in RAGU allows you to change instructions.
Prompts are defined by the `RAGUInstruction` class.

```python
@dataclass
class PromptTemplate:
    """
    Represents a Jinja2-based prompt template for instruction generation.

    Each template defines:
      - a Jinja2 text pattern (`template`)
      - an optional Pydantic schema for structured output validation (`schema`)
      - a short description of its purpose (`description`)

    The template can be rendered dynamically with keyword arguments,
    supporting both single-instance and batched (list/tuple) generation.
    """

    template: str                                           # Jinja2 template
    pydantic_model: Type[BaseModel] | Type[str] = str       # Pydantic schema
    description: str = ""                                   # Short instruction description
```

Retrieve all available instructions:

```python
from ragu import LocalSearchEngine

search_engine = LocalSearchEngine(
    llm=llm,
    knowledge_graph=knowledge_graph,
    embedder=embedder,
)

all_prompts = search_engine.get_prompts()
print(all_prompts)

local_search_prompt = search_engine.get_prompt("local_search")
print(local_search_prompt.messages.to_str())
```

Update a specific instruction:

```python
from textwrap import dedent

from ragu.common.prompts.messages import ChatMessages, SystemMessage, UserMessage
from ragu.common.prompts.prompt_storage import RAGUInstruction

search_engine.update_prompt(
    "local_search",
    RAGUInstruction(
        messages=ChatMessages.from_messages([
            SystemMessage(content="You answer using only the supplied graph context."),
            UserMessage(content=dedent(
                """
                Query: {{ query }}
                Context: {{ context }}
                Language: {{ language }}
                """
            )),
        ]),
        pydantic_model=str,
        description="Custom local-search instruction",
    ),
)
```
