
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

### Conceptual documentation

- [RAGU components and methodology (EN)](docs/en/ragu_components.md) | [RU](docs/ru/ragu_components.md)
- [NEREL ontology (EN)](docs/en/ontology.md) | [RU](docs/ru/ontology.md)
- [RAGU-lm extraction model (EN)](docs/en/ragu_lm.md) | [RU](docs/ru/ragu_lm.md)

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
The recommended way is a local build:
```commandline
git clone https://github.com/AsphodelRem/RAGU.git
cd RAGU
uv pip install -e .
```

From PyPI:
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

RAGU ships four retrieval strategies plus a query-planning wrapper. Each engine
exposes `a_search()` (retrieval only) and `a_query()` (retrieval + LLM answer);
sync wrappers `search()` / `query()` are also available. For the conceptual
workflow of each strategy see [`docs/en/ragu_components.md`](docs/en/ragu_components.md);
for full parameters see [`ragu/search_engine/README.md`](ragu/search_engine/README.md).

#### Local search

Graph-neighborhood retrieval: find relevant entities, then expand to their relations, community summaries, and source chunks.

```python
from ragu import LocalSearchEngine

local_search = LocalSearchEngine(llm=llm, knowledge_graph=knowledge_graph, embedder=embedder)
local_answer = await local_search.a_query("Who wrote Romeo and Juliet?", use_summary=True, use_chunks=True)
print(local_answer.response)
```

#### Global search

Answers broad, corpus-wide questions from community summaries.

```python
from ragu import GlobalSearchEngine

global_search = GlobalSearchEngine(llm=llm, knowledge_graph=knowledge_graph)
global_answer = await global_search.a_query("Your broad query here")
print(global_answer.response)
```

#### Naive search (vector RAG)

Chunk-vector RAG without graph expansion.

```python
from ragu import NaiveSearchEngine

naive_search = NaiveSearchEngine(llm=llm, knowledge_graph=knowledge_graph, embedder=embedder)
naive_answer = await naive_search.a_query("Your query here")
print(naive_answer.response)
```

#### Mixed search

Runs several engines and asks the LLM to synthesize a single answer from their responses.

```python
from ragu import MixSearchEngine

mix_search = MixSearchEngine(llm=llm, engines=[local_search, naive_search, global_search])
mixed_answer = await mix_search.a_query("Your query here")
print(mixed_answer.response)
```

#### Query planning wrapper

Wraps any engine, decomposes a complex question into dependent subqueries, and feeds intermediate answers forward.

```python
from ragu import QueryPlanEngine

planned_local = QueryPlanEngine(local_search)
result = await planned_local.a_query("Who wrote the novel 'Quo Vadis' and what country was the author from?")
print(result.response)
```

---

### Advanced Configuration

#### Builder Settings

Configure the knowledge graph building pipeline using `BuilderArguments`:

```python
from ragu import BuilderArguments, KnowledgeGraph

builder_arguments = BuilderArguments(
    use_llm_summarization=True,  # Enable LLM-based entity/relation summarization
    use_clustering=False,  # Apply clustering before summarization. Use it if your text contains many similar entities.
    build_only_vector_context=False,  # Skip graph extraction, only chunk embeddings
    make_community_summary=True,  # Generate community summaries
    remove_isolated_nodes=True,  # Remove entities without relations
    cluster_only_if_more_than=10000,  # Minimum entities before clustering kicks in
    summarize_only_if_more_than=7,  # Summarize descriptions only when there are many duplicates
    max_cluster_size=128,  # Maximum entities per cluster
    random_seed=42,
)

knowledge_graph = KnowledgeGraph(
    llm=llm,
    embedder=embedder,
    chunker=chunker,
    artifact_extractor=artifact_extractor,
    builder_settings=builder_arguments,
)
await knowledge_graph.build_from_docs(docs)
```

Common presets (naive vector RAG only, fast graph extraction without
summarization, full GraphRAG with communities, and large-corpus clustering) with
explanations and ready-to-run scripts are in
[`ragu/graph/README.md`](ragu/graph/README.md).

---

#### Client and Rate Limiting Configuration

`CachedAsyncOpenAI` is the network-level client shared by `LLMOpenAI` and `EmbedderOpenAI`. It controls rate limiting, retries, caching, and timeouts.

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

For the full parameter tables (`rate_*`, `retry_times_sec`, `embed_timeout`, `EmbedderOpenAI` batching and tokenizer overrides), the **shared vs. separate clients** guidance for large corpora, and the **debug store**, see [`ragu/models/README.md`](ragu/models/README.md).

---

#### Token Limits and Settings

RAGU centralises token-limit and tokenizer configuration in the `Settings` singleton. These defaults are used by `EmbedderOpenAI` (for embedding input truncation) and search engines (for LLM context truncation).

```python
from ragu import Settings

# Embedder truncation (applied automatically inside EmbedderOpenAI)
Settings.embedder_token_limit = 8_192                    # max tokens per embedding input
Settings.tokenizer_embedder_backend = "tiktoken"         # "tiktoken" or "local"
Settings.tokenizer_embedder_name = "text-embedding-3-large"

# LLM context truncation (applied by search engines before answer generation)
Settings.llm_context_token_limit = 30_000                # max tokens for search-engine context
Settings.tokenizer_llm_backend = "tiktoken"              # "tiktoken" or "local"
Settings.tokenizer_llm_name = "gpt-4o"
```

Per-instance overrides on `EmbedderOpenAI` take precedence over `Settings`.

For a local BGE model with a 512-token context, set
`Settings.embedder_token_limit = 512`, `Settings.tokenizer_embedder_backend = "local"`,
and `Settings.tokenizer_embedder_name = "BAAI/bge-large-en-v1.5"` (requires
`pip install graph_ragu[local]`).

---

#### Serializing Global Settings

`Settings.save(path)` writes a JSON snapshot of the configuration;
`Settings.load(path)` restores it. Serialization is never invoked automatically.
For the list of serialized/excluded fields and the validation behavior, see
[`ragu/common/README.md`](ragu/common/README.md).

```python
from ragu import Settings

Settings.save("./runs/exp_42/ragu_settings.json")   # persist
Settings.load("./runs/exp_42/ragu_settings.json")   # restore in a fresh process
```

---

#### In-Context Learning (Few-Shot Examples)

RAGU extractors can use few-shot examples to improve extraction quality. When enabled, the extractor selects relevant examples and includes them in the LLM prompt.

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

Four selection strategies are available (`"semantic"`, `"bm25"`, `"hybrid"`, `"random"`). Pre-built example files ship in `ragu/common/prompts/icl_examples/`; to generate custom ones, run `python scripts/generate_icl_examples.py --config config/icl_generation.yaml`. For the strategy comparison table and per-strategy embedder requirements, see [`ragu/common/prompts/README.md`](ragu/common/prompts/README.md).

---


### Knowledge Graph Construction
Each text in the corpus is processed to extract structured information. It consists of:

* **Entities** — textual representation, entity type, and a contextual description.
* **Relations** — textual description of the link between two entities (or a relation class), as well as its confidence/strength.

> **RAGU uses entity and relation classes from [NEREL](https://github.com/nerel-ds/NEREL).** The full type tables are available in English and Russian: [`docs/en/ontology.md`](docs/en/ontology.md) | [`docs/ru/ontology.md`](docs/ru/ontology.md). Pass custom `entity_types` / `relation_types` to the extractors to override the defaults.
>
> `Entity` and `Relation` are base graph model classes. They can be inherited to create richer domain-specific node and edge types, provided the storage `Node` / `Edge` contract is preserved. See [graph docs](ragu/graph/README.md) and [storage docs](ragu/storage/README.md).

### Extraction pipelines

#### 1. Single-step LLM pipeline

File: `ragu/triplet/llm_artifact_extractor.py`.
A baseline pipeline that uses an LLM to extract entities, relations, and their descriptions in a single step. Supports optional in-context learning with few-shot examples and artifact validation.

#### 2. Two-stage LLM pipeline

File: `ragu/triplet/two_stage_extractor.py`.
Extracts entities first, then extracts relations constrained by the entity list. It can separately validate entity and relation outputs. Supports optional in-context learning with few-shot examples.

#### 3. RAGU-lm (Russian language)

A compact model (Qwen-3-0.6B) fine-tuned on the NEREL dataset. Recognizes and normalizes entities, generates descriptions, and extracts relations. Full prompts, a worked example, and an F1 comparison live in [`docs/en/ragu_lm.md`](docs/en/ragu_lm.md) (RU: [`docs/ru/ragu_lm.md`](docs/ru/ragu_lm.md)).

---

### Prompt Customization

All RAGU components that use LLMs inherit from `RaguGenerativeModule`, which provides `get_prompt`, `get_prompts`, and `update_prompt` to view and override instructions.

```python
from ragu import LocalSearchEngine

search_engine = LocalSearchEngine(
    llm=llm,
    knowledge_graph=knowledge_graph,
    embedder=embedder,
)

all_prompts = search_engine.get_prompts()           # {'local_search': RAGUInstruction(...)}
local_search_prompt = search_engine.get_prompt("local_search")
print(local_search_prompt.messages.to_str())        # rendered prompt text
print(local_search_prompt.pydantic_model)           # response schema
```

To override an instruction, build a new `RAGUInstruction` and call
`search_engine.update_prompt("local_search", custom_instruction)`. Do **not**
edit `DEFAULT_PROMPT_TEMPLATES` directly — `update_prompt` scopes the change to a
single module instance. Full examples (including custom messages and few-shot
formatters) are in [`ragu/common/prompts/README.md`](ragu/common/prompts/README.md).
---

### Contributors
#### **Main Idea & Inspiration**
- Ivan Bondarenko — idea, smart_chunker, NER model, ragu-lm


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
