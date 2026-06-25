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

In the standard extraction pipeline, NEREL entity and relation types are injected into extraction prompts by default. You can pass custom `entity_types` and `relation_types` to the LLM extractors when you need a different ontology.

> The full list of entity and relation types is in [`ontology.md`](ontology.md).

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
3. **lightweight fine-tuned model approach** with [RAGU-lm](https://huggingface.co/RaguTeam/RAGU-lm) (only for Russian language)

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

To enable response caching for every client without repeating `cache=...`, set `Settings.cache_path` once (see `ragu/models/README.md` and `ragu/common/README.md`). The path must be a stable directory **independent of `Settings.storage_folder`** (which is per-run); mind stale hits when changing model, temperature, or provider.

Single-step extraction and validation are driven by the `artifact_extraction` and `artifact_validation` prompts. Two-stage extraction uses `entity_extraction`, `entity_validation`, `relation_extraction`, and `relation_validation`.

#### RAGU-lm

A lightweight fine-tuned model approach for Russian-language extraction, based on [RAGU-lm](https://huggingface.co/RaguTeam/RAGU-lm) (Qwen-3-0.6B). It recognizes and normalizes named entities, generates entity descriptions, and extracts directed relations between entity pairs.

The full prompt set, a worked example, a comparison table, and usage instructions live in [`ragu_lm.md`](ragu_lm.md).

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

Every LLM-powered component in RAGU inherits `RaguGenerativeModule` and allows
you to inspect and override instructions. Instructions are defined by the
`RAGUInstruction` class, which binds a Jinja2-templated conversation
(`messages: ChatMessages`) with an optional Pydantic schema (`pydantic_model`),
a short `description`, and an optional `few_shot_formatter`.

Retrieve the instructions used by a component:

```python
from ragu import LocalSearchEngine

search_engine = LocalSearchEngine(
    llm=llm,
    knowledge_graph=knowledge_graph,
    embedder=embedder,
)

all_prompts = search_engine.get_prompts()             # name -> RAGUInstruction
local_search_prompt = search_engine.get_prompt("local_search")
print(local_search_prompt.messages.to_str())
```

Override an instruction with `search_engine.update_prompt(name, instruction)`.
Do **not** edit `DEFAULT_PROMPT_TEMPLATES` directly — `update_prompt` scopes the
change to a single instance. Full field reference, custom-message examples, and
few-shot formatting are in
[`../../ragu/common/prompts/README.md`](../../ragu/common/prompts/README.md).
```
