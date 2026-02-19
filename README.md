
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
RAGU provides a pipeline for building a **Knowledge Graph**, and performing retrieve over the indexed data. It contains different approaches to extract structured data from raw texts to enable efficient question-answering over structured knowledge.

Partially based on [nano-graphrag](https://github.com/gusye1234/nano-graphrag/tree/main)

Our huggingface community is [here](https://huggingface.co/RaguTeam/)

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

If you want to use local models (via transformers etc), run:
```bash
pip install graph_ragu[local]
```

---

## Quickstart

### Simple example of building knowledge graph

```python
import asyncio

from openai import AsyncOpenAI
from ragu import (
    SimpleChunker,
    KnowledgeGraph,
    BuilderArguments,
    Settings,
    ArtifactsExtractorLLM,
)
from ragu.llm import OpenAIClient
from ragu.embedder import OpenAIEmbedder

from ragu.utils.ragu_utils import read_text_from_files

# Configuration (or use ragu.Env for loading from .env)
LLM_MODEL_NAME = "openai/gpt-4o-mini"
LLM_BASE_URL = "https://api.openai.com/v1"
LLM_API_KEY = "your-api-key-here"

EMBEDDER_MODEL_NAME = "text-embedding-3-large"
EMBEDDER_DIM = 3072


async def main():
    # Configure working directory and language
    Settings.storage_folder = "ragu_working_dir"
    Settings.language = "english"  # or "russian"

    # Load documents from folder
    docs = read_text_from_files("path/to/your/data")

    # Initialize chunker
    chunker = SimpleChunker(max_chunk_size=1000)

    # Set up LLM client
    client = OpenAIClient(
        model_name=LLM_MODEL_NAME,
        base_url=LLM_BASE_URL,
        api_token=LLM_API_KEY,
        max_requests_per_second=1,
        max_requests_per_minute=60,
        cache_flush_every=10,
    )

    # Set up artifact extractor
    artifact_extractor = ArtifactsExtractorLLM(
        client=client,
        do_validation=False
    )

    # Initialize embedder
    embedder = OpenAIEmbedder(
        model_name=EMBEDDER_MODEL_NAME,
        client=AsyncOpenAI(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
        ),
        dim=EMBEDDER_DIM,
        max_requests_per_second=1,
        max_requests_per_minute=60,
        cache_dir='cache/',
    )

    # Configure builder settings
    builder_settings = BuilderArguments(
        use_llm_summarization=True,
        vectorize_chunks=True,
    )

    # Build knowledge graph
    knowledge_graph = await KnowledgeGraph(
        client=client,
        embedder=embedder,
        chunker=chunker,
        artifact_extractor=artifact_extractor,
        builder_settings=builder_settings,
    ).build_from_docs(docs)


if __name__ == "__main__":
    asyncio.run(main())
```

> If you run the code with a storage folder that already contains a knowledge graph, RAGU will automatically load the existing graph.


### Example of querying

**Local search**
Search over entities retrieved for the query and their connected context (relations, summaries, and chunks).
```python
from ragu import LocalSearchEngine

local_search = LocalSearchEngine(
    client,
    knowledge_graph,
    embedder,
    tokenizer_model="gpt-4o-mini",
)
local_answer = await local_search.a_query("Who wrote Romeo and Juliet?")
print(local_answer)
```

#### Global search
Give an answer by community summaries.
```python
from ragu import GlobalSearchEngine

global_search = GlobalSearchEngine(
    client=client,
    knowledge_graph=knowledge_graph,
)
global_answer = await global_search.a_query("Your broad query here")
print(global_answer)
```

**Naive search (vector RAG):**
```python
from ragu import NaiveSearchEngine

naive_search = NaiveSearchEngine(
    client=client,
    knowledge_graph=knowledge_graph,
    embedder=embedder,
)
naive_answer = await naive_search.a_query("Your query here")
print(naive_answer)
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

Configure the knowledge graph building pipeline using `BuilderSettings`:

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
    max_cluster_size=128,  # Maximum entities per cluster
)

# Pass to KnowledgeGraph
knowledge_graph = await KnowledgeGraph(
    client=client,
    embedder=embedder,
    chunker=chunker,
    artifact_extractor=artifact_extractor,
    builder_settings=builder_arguments,
).build_from_docs(docs)
```
---

### Knowledge Graph Construction
Each text in corpus is processed to extract structured information. It consist of:

* **Entities** — textual representation, entity type, and a contextual description.
* **Relations** — textual description of the link between two entities (or a relation class), as well as its confidence/strength.

> **RAGU uses entity and relation classes from [NEREL](https://github.com/nerel-ds/NEREL).**

### Entity types

|No. | Entity type | No. | Entity type | No. | Entity type |
|---|---|---|---|---|---|
|1.| AGE |11.| FAMILY |21.| PENALTY |
|2.| AWARD |12.| IDEOLOGY |22.| PERCENT |
|3.| CITY |13.| LANGUAGE |23.| PERSON |
|4.| COUNTRY |14.| LAW |24.| PRODUCT |
|5.| CRIME |15.| LOCATION |25.| PROFESSION |
|6.| DATE |16.| MONEY |26.| RELIGION |
|7.| DISEASE |17.| NATIONALITY |27.| STATE_OR_PROV |
|8.| DISTRICT |18.| NUMBER |28.| TIME |
|9.| EVENT |19.| ORDINAL |29.| WORK_OF_ART |
|10.| FACILITY |20.| ORGANIZATION | | |

### Relation types

|No. | Relation type | No. | Relation type | No. | Relation type |
|---|---|---|---|---|---|
|1.| ABBREVIATION |18.| HEADQUARTERED_IN |35.| PLACE_RESIDES_IN |
|2.| AGE_DIED_AT |19.| IDEOLOGY_OF |36.| POINT_IN_TIME |
|3.| AGE_IS |20.| INANIMATE_INVOLVED |37.| PRICE_OF |
|4.| AGENT |21.| INCOME |38.| PRODUCES |
|5.| ALTERNATIVE_NAME |22.| KNOWS |39.| RELATIVE |
|6.| AWARDED_WITH |23.| LOCATED_IN |40.| RELIGION_OF |
|7.| CAUSE_OF_DEATH |24.| MEDICAL_CONDITION |41.| SCHOOLS_ATTENDED |
|8.| CONVICTED_OF |25.| MEMBER_OF |42.| SIBLING |
|9.| DATE_DEFUNCT_IN |26.| ORGANIZES |43.| SPOUSE |
|10.| DATE_FOUNDED_IN |27.| ORIGINS_FROM |44.| START_TIME |
|11.| DATE_OF_BIRTH |28.| OWNER_OF |45.| SUBEVENT_OF |
|12.| DATE_OF_CREATION |29.| PARENT_OF |46.| SUBORDINATE_OF |
|13.| DATE_OF_DEATH |30.| PART_OF |47.| TAKES_PLACE_IN |
|14.| END_TIME |31.| PARTICIPANT_IN |48.| WORKPLACE |
|15.| EXPENDITURE |32.| PENALIZED_AS |49.| WORKS_AS |
|16.| FOUNDED_BY |33.| PLACE_OF_BIRTH | | |
|17.| HAS_CAUSE |34.| PLACE_OF_DEATH | | |


### How it is extracted:
#### 1. Default Pipeline

File: ragu/triplet/llm_artifact_extractor.py.
A baseline pipeline that uses LLM to extract entities, relations, and their descriptions in a single step.

#### 2. [RAGU-lm](https://huggingface.co/RaguTeam/RAGU-lm) (for russian language)
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
    client,
    knowledge_graph,
    embedder
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
from ragu.common.prompts.default_models import DefaultResponseModel

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
    pydantic_model=DefaultResponseModel,  # Your pydantic model (optional)
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

