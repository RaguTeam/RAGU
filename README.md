
<h1 align="center">RAGU: Retrieval-Augmented Graph Utility</h1>

---

<p align="center">
<img src="assets/ragu_image.jpg" alt="RAGU logo" width="600" />
</p>

<h4 align="center">
  <a href="https://github.com/AsphodelRem/RAGU/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="RAGU is under the MIT license." alt="RAGU"/>
  </a>
  <img src="https://img.shields.io/badge/python->=3.10-blue">
</h4>

<h4 align="center">
  <a href="#install">Install</a> |
  <a href="#quickstart">Quickstart</a> 
</h4>

---


## Overview
RAGU is a modular GraphRAG engine for building, storing, and querying knowledge graphs from text. It combines:

- chunking of raw documents into stable `Chunk` objects;
- LLM-based extraction of `Entity` and `Relation` graph artifacts;
- graph construction, deduplication, optional description summarization, and Leiden community detection;
- graph, key-value, and vector storage backends;
- retrieval strategies for local graph neighborhoods, global community summaries, naive chunk-vector RAG, and mixed search.

Partially based on [nano-graphrag](https://github.com/gusye1234/nano-graphrag/tree/main)

Our huggingface community is [here](https://huggingface.co/RaguTeam/)

### Module documentation

- [Package facade](ragu/README.md)
- [Chunking](ragu/chunker/README.md)
- [Common settings, prompts, and utilities](ragu/common/README.md)
- [Prompt schemas and templates](ragu/common/prompts/README.md)
- [Graph construction and index](ragu/graph/README.md)
- [Models, embedders, sparse embedders, and rerankers](ragu/models/README.md)
- [Search engines](ragu/search_engine/README.md)
- [Storage contracts and adapters](ragu/storage/README.md)
- [Graph storage adapters](ragu/storage/graph_storage_adapters/README.md)
- [Vector DB adapters](ragu/storage/vdb_storage_adapters/README.md)
- [Triplet/entity-relation extraction](ragu/triplet/README.md)
- [Utilities](ragu/utils/README.md)

---

## Install
Better way is a local build:
```commandline
git clone https://github.com/AsphodelRem/RAGU.git
cd RAGU
uv pip install -e .
```

From pypi:
```bash
pip install graph_ragu
```

If you want to use local models (via transformers etc.), run:
```bash
pip install graph_ragu[local]
```

---

## Quickstart

### Simple example of building knowledge graph

```python
import asyncio
import os
import sys
import shutil

from ragu.common.logger import logger
logger.remove()
logger.add(sys.stdout, level="DEBUG") 

from ragu import (
    SimpleChunker,
    KnowledgeGraph,
    BuilderArguments,
    Settings,
    TwoStageArtifactsExtractorLLM,
)
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI
from ragu.utils.ragu_utils import read_text_from_files

client = CachedAsyncOpenAI(
    base_url=os.environ['OPENAI_BASE_URL'],
    api_key=os.environ['OPENAI_API_KEY'],
    rate_min_delay=2,
    rate_max_simultaneous=10,
    retry_times_sec=(2, 2, 2, 2, 2),
    cache='./llm_cache',
    debug_errors_storage='./llm_debug',
)

llm = LLMOpenAI(
    client=client,
    model_name="mistralai/mistral-medium-3",
)
embedder = EmbedderOpenAI(
    client=client,
    model_name="emb-qwen/qwen3-embedding-8b",
    dim=4096,
)

# Configure working directory and language
Settings.storage_folder = "ragu_working_dir"
Settings.language = "english"  # or "russian"

# Remove dir to start building graph from scratch
# shutil.rmtree(Settings.storage_folder, ignore_errors=True)

docs = read_text_from_files("path/to/your/files")

# Initialize chunker
chunker = SimpleChunker(max_chunk_size=1000)

# Set up artifact extractor
from ragu.common.prompts import ICLConfig

icl_config = ICLConfig(
    enabled=True,
    num_examples=2,
    language="english",
)

artifact_extractor = TwoStageArtifactsExtractorLLM(
    llm=llm,
    embedder=embedder,
    icl_config=icl_config,
    do_entity_validation=True,
    do_relation_validation=True,
)

# Configure builder settings
builder_settings = BuilderArguments(
    use_llm_summarization=True,
    use_clustering=False,
    build_only_vector_context=False,
    make_community_summary=True,
    remove_isolated_nodes=True,
)

# Build knowledge graph
knowledge_graph = KnowledgeGraph(
    llm=llm,
    embedder=embedder,
    chunker=chunker,
    artifact_extractor=artifact_extractor,
    builder_settings=builder_settings,
)

asyncio.run(knowledge_graph.build_from_docs(docs))
```

> If you run the code with a storage folder that already contains a knowledge graph, RAGU will automatically load the existing graph.


### Example of querying

**Local search**
Search over entities retrieved for the query and their connected context (relations, summaries, and chunks).
```python
from ragu.search_engine.local_search import LocalSearchEngine

local_search = LocalSearchEngine(
    llm=llm,  # or use another LLM for answering
    knowledge_graph=knowledge_graph,
    embedder=embedder,
    tokenizer_model="gpt-4o-mini",
)
# found = await local_search.a_search("What is the Betweenlands??")
local_answer = await local_search.a_query(
    "Who wrote Romeo and Juliet?",
    use_summary=True,
    use_chunks=True,
)
print(local_answer.response)
```

#### Global search
Give an answer by community summaries.
```python
from ragu import GlobalSearchEngine

global_search = GlobalSearchEngine(
    llm=llm,
    knowledge_graph=knowledge_graph,
)
global_answer = await global_search.a_query("Your broad query here")
print(global_answer.response)
```

**Naive search (vector RAG):**
```python
from ragu import NaiveSearchEngine

naive_search = NaiveSearchEngine(
    llm=llm,
    knowledge_graph=knowledge_graph,
    embedder=embedder,
)
naive_answer = await naive_search.a_query("Your query here")
print(naive_answer.response)
```

**Mixed search:**
```python
from ragu import MixSearchEngine

mix_search = MixSearchEngine(
    llm=llm,
    engines=[local_search, naive_search, global_search],
)
mixed_answer = await mix_search.a_query("Your query here")
print(mixed_answer.response)
```

### Query planning wrapper
Decomposes complex questions into dependent subqueries, executes them in order, and uses intermediate answers to produce a final response.
```python
from ragu import QueryPlanEngine

# Wrap any base engine
planned_local = QueryPlanEngine(local_search)
result = await planned_local.a_query("What is the capital of France?")
print(result)

planned_global = QueryPlanEngine(global_search)
result = await planned_global.a_query("Your broad query here")
print(result)

planned_naive = QueryPlanEngine(naive_search)
result = await planned_naive.a_query("Your query here")
print(result)
```

---

### Advanced Configuration

#### Builder Settings

Configure the knowledge graph building pipeline using `BuilderArguments`:

```python
from ragu import BuilderArguments

builder_arguments = BuilderArguments(
    use_llm_summarization=True,  # Enable LLM-based entity/relation summarization
    use_clustering=False,  # Apply clustering before summarization. Use it if your text contains many similar entities.
    build_only_vector_context=False,  # Skip graph extraction, only chunk embeddings
    make_community_summary=True,  # Generate community summaries 
    remove_isolated_nodes=True,  # Remove entities without relations
    vectorize_chunks=True,  # Vectorize chunk for naive (vector) search
    cluster_only_if_more_than=10000,  # Minimum entities before clustering kicks in
    summarize_only_if_more_than=7,  # Summarize descriptions only when there are many duplicates
    max_cluster_size=128,  # Maximum entities per cluster
    random_seed=42,
)

# Pass to KnowledgeGraph
knowledge_graph = KnowledgeGraph(
    llm=llm,
    embedder=embedder,
    chunker=chunker,
    artifact_extractor=artifact_extractor,
    builder_settings=builder_arguments,
)
await knowledge_graph.build_from_docs(docs)
```

Common builder presets:

```python
# Naive vector RAG only: chunks + chunk vectors, no entity/relation graph.
BuilderArguments(
    build_only_vector_context=True,
    make_community_summary=False,
)

# Fast graph extraction: no extra LLM summarization and no communities.
BuilderArguments(
    use_llm_summarization=False,
    use_clustering=False,
    make_community_summary=False,
    remove_isolated_nodes=True,
)

# Full GraphRAG: extraction, summarization, communities, global search support.
BuilderArguments(
    use_llm_summarization=True,
    use_clustering=False,
    make_community_summary=True,
    remove_isolated_nodes=True,
)
```

---

#### Client and Rate Limiting Configuration

`CachedAsyncOpenAI` is the network-level client shared by `LLMOpenAI` and `EmbedderOpenAI`. It controls rate limiting, retries, caching, and timeouts.

**Shared client (simpler, works for most cases):**

```python
client = CachedAsyncOpenAI(
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    rate_max_simultaneous=10,
    rate_max_per_minute=100,
    cache="./llm_cache",
)

llm = LLMOpenAI(client=client, model_name="gpt-4o-mini")
embedder = EmbedderOpenAI(client=client, model_name="text-embedding-3-large", dim=3072)
```

**Separate clients (recommended for large corpora or different providers):**

When processing large corpora (thousands of entities and relations), LLM calls (slow, seconds per request) and embedding calls (fast but numerous) compete for the same connection pool and rate limiter. Using separate clients isolates their resources:

```python
llm_client = CachedAsyncOpenAI(
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    rate_max_simultaneous=5,
    rate_max_per_minute=60,
    retry_times_sec=(4, 8, 16),
    cache="./llm_cache",
)

embed_client = CachedAsyncOpenAI(
    base_url="https://api.openai.com/v1",  # can be a different provider
    api_key="sk-...",
    rate_max_simultaneous=20,
    rate_max_per_minute=500,
    embed_timeout=60.0,
    cache="./embed_cache",
)

llm = LLMOpenAI(client=llm_client, model_name="gpt-4o-mini")
embedder = EmbedderOpenAI(
    client=embed_client,
    model_name="text-embedding-3-large",
    dim=3072,
    batch_size=500,
    max_concurrent_batches=5,
)
```

**When a shared client is sufficient:**
- Small to medium documents (up to ~1000 entities/relations)
- Generous API provider rate limits
- Both LLM and embedder use the same endpoint

**When separate clients are recommended:**
- Large corpora (thousands of entities and relations)
- Strict API rate limits (low RPM)
- LLM and embedder on different providers or endpoints
- You need independent scaling of LLM vs. embedding throughput

**Key parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `rate_max_simultaneous` | Max concurrent in-flight requests | None (unlimited) |
| `rate_max_per_minute` | Max requests per minute | None (unlimited) |
| `rate_min_delay` | Min seconds between request starts | None |
| `retry_times_sec` | Retry wait schedule on transient errors | `(2, 4, 8)` |
| `embed_timeout` | Per-request timeout for embedding calls (seconds) | `60.0` |
| `cache` | Cache directory path | None |

`EmbedderOpenAI` also accepts:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `batch_size` | Max texts per single API call | `500` |
| `max_concurrent_batches` | Max concurrent batch API calls | `5` |

---

#### In-Context Learning (Few-Shot Examples)

RAGU extractors can use few-shot examples to improve extraction quality. When enabled, the extractor selects relevant examples and includes them in the LLM prompt. Four selection strategies are available:

| Strategy | Description | Requires `embedder` |
|----------|-------------|:---:|
| `"semantic"` | Cosine similarity on dense embeddings (default) | Yes |
| `"bm25"` | Lexical matching via BM25 — fast, no API calls | No |
| `"hybrid"` | Reciprocal Rank Fusion of semantic + BM25 rankings | Yes |
| `"random"` | Uniform random sampling — useful as a baseline | No |

```python
from ragu.common.prompts import ICLConfig

icl_config = ICLConfig(
    enabled=True,                       # Enable/disable ICL
    num_examples=2,                     # Number of examples per extraction call (1-3 recommended)
    selection_strategy="semantic",      # "semantic" | "bm25" | "hybrid" | "random"
)

artifact_extractor = TwoStageArtifactsExtractorLLM(
    llm=llm,
    embedder=embedder,                  # Required for "semantic" and "hybrid"; optional for "bm25" and "random"
    icl_config=icl_config,
)
```

Both `ArtifactsExtractorLLM` and `TwoStageArtifactsExtractorLLM` support ICL. An `embedder` is required for `"semantic"` and `"hybrid"` strategies; for `"bm25"` and `"random"` it can be omitted. Pre-built example files ship with RAGU in `ragu/common/prompts/icl_examples/`.

To generate custom examples for your domain, use the generation script:

```bash
python scripts/generate_icl_examples.py --config config/icl_generation.yaml
```

---


### Knowledge Graph Construction
Each text in corpus is processed to extract structured information. It consist of:

* **Entities** — textual representation, entity type, and a contextual description.
* **Relations** — textual description of the link between two entities (or a relation class), as well as its confidence/strength.

> **RAGU uses entity and relation classes from [NEREL](https://github.com/nerel-ds/NEREL).**
>
> `Entity` and `Relation` are base graph model classes. They can be inherited to create richer domain-specific node and edge types, provided the storage `Node` / `Edge` contract is preserved. See [graph docs](ragu/graph/README.md) and [storage docs](ragu/storage/README.md).

### Entity types

| No. | Entity type | No. | Entity type  | No. | Entity type   |
|-----|-------------|-----|--------------|-----|---------------|
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

### Relation types

| No. | Relation type    | No. | Relation type      | No. | Relation type    |
|-----|------------------|-----|--------------------|-----|------------------|
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


### How it is extracted:
#### 1. Default Pipeline

File: ragu/triplet/llm_artifact_extractor.py.
A baseline pipeline that uses LLM to extract entities, relations, and their descriptions in a single step. Supports optional in-context learning with few-shot examples and artifact validation.

#### 2. Two-stage LLM Pipeline

File: ragu/triplet/two_stage_extractor.py.
Extracts entities first, then extracts relations constrained by the entity list. It can separately validate entity and relation outputs. Supports optional in-context learning with few-shot examples.

#### 3. [RAGU-lm](https://huggingface.co/RaguTeam/RAGU-lm) (for russian language)
A compact model (Qwen-3-0.6B) fine-tuned on the NEREL dataset.
The pipeline operates in several stages:
1. Extract unnormalized entities from text.
2. Normalize entities into canonical forms.
3. Generate entity descriptions.
4. Extract relations based on the inner product between entities.

### Comparison
| Model                 | Dataset | F1 (Entities) | F1 (Relations) |
|-----------------------|----------|---------------|----------------|
| Qwen-2.5-14B-Instruct | NEREL | 0.32          | 0.69           |
| RAGU-lm (Qwen-3-0.6B) | NEREL | 0.6           | 0.71           |
| Small-model pipeline  | NEREL | 0.74          | 0.75           |

---

### Prompt Customization

All RAGU components that use LLMs inherit from `RaguGenerativeModule`, which provides methods to view and update prompts.

#### Viewing Current Prompts

```python
from ragu import LocalSearchEngine

search_engine = LocalSearchEngine(
    llm=llm,
    knowledge_graph=knowledge_graph,
    embedder=embedder,
)

# Get all prompts used by the search engine
all_prompts = search_engine.get_prompts()
print(all_prompts)
# Returns: {'local_search': RAGUInstruction(...)}

# Get a specific prompt
local_search_prompt = search_engine.get_prompt("local_search")
print(local_search_prompt.messages.to_str())
# Shows the actual prompt content (all conversation as single text)

print(local_search_prompt.pydantic_model)
# Shows the response pydantic model 
```

#### Updating Prompts

You can customize prompts by creating a new `RAGUInstruction` with your own messages:

```python
from textwrap import dedent

from ragu.common.prompts.prompt_storage import RAGUInstruction
from ragu.common.prompts.messages import ChatMessages, UserMessage, SystemMessage

# Create custom prompt instruction
custom_instruction = RAGUInstruction(
    messages=ChatMessages.from_messages([
        SystemMessage(content="You are a helpful assistant specialized in academic research."),
        UserMessage(content=dedent(
            """
            Answer the following query using the provided context.
            
            Query: {{ query }}
            Context: {{ context }}
            
            Language: {{ language }}
            """
        ))  # Can store any conversation
    ]),
    pydantic_model=str,  # Or your own pydantic BaseModel subclass
    description="Custom local search prompt with academic focus" # Optional
)

# Update the prompt
search_engine.update_prompt("local_search", custom_instruction)
```
---

### Contributors
#### **Main Idea & Inspiration**
- Ivan Bondarenko - idea, smart_chunker, NER model, ragu-lm


#### **Core Development**

- Mikhail Komarov

#### **Benchmarks & Evaluation**  
- Roman Shuvalov
- Yanya Dement'yeva 
- Alexandr Kuleshevskiy
- Nikita Kukuzey
- Stanislav Shtuka

#### **Small Models Pipeline**
- Matvey Solovyev
- Ilya Myznikov
