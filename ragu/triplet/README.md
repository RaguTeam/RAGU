# Module: ragu.triplet

## Role in RAGU Pipeline

`ragu.triplet` is the extraction layer. It converts chunks into graph artifacts: entities and relations. These artifacts are then summarized, clustered, and stored by `ragu.graph`.

Pipeline position:

```text
List[Chunk] -> artifact extractor -> List[Entity], List[Relation] -> graph builder
```

## Overview

The module exists to isolate LLM prompting and structured artifact conversion from graph storage. Extractors use prompts and Pydantic schemas to get structured output, then convert that output into `Entity` and `Relation` dataclasses with stable IDs and source chunk references.

## Key Components

### BaseArtifactExtractor

Abstract extractor interface.

- Purpose: standardize `extract(chunks) -> (entities, relations)`.
- Important behavior: instances are callable as async functions.
- Used by: `InMemoryGraphBuilder.extract_graph`.

```python
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI
from ragu.triplet import ArtifactsExtractorLLM

client = CachedAsyncOpenAI(base_url="https://api.openai.com/v1", api_key="dummy-api-token")
llm = LLMOpenAI(client=client, model_name="gpt-4o-mini")
extractor = ArtifactsExtractorLLM(llm)

print(extractor.get_prompt("artifact_extraction").description)
```

### ArtifactsExtractorLLM

Single-pass LLM extractor.

- Purpose: extract entities and relations from each chunk in one structured call.
- Optional validation: `do_validation=True` runs an additional artifact validation prompt.
- Important parameters: `llm`, `language`, `entity_types`, `relation_types`.

```python
import asyncio

from ragu.chunker.types import Chunk
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI
from ragu.triplet import ArtifactsExtractorLLM


async def main():
    client = CachedAsyncOpenAI(base_url="https://api.openai.com/v1", api_key="dummy-api-token")
    llm = LLMOpenAI(client=client, model_name="gpt-4o-mini")
    extractor = ArtifactsExtractorLLM(llm, do_validation=False)
    chunks = [Chunk("Python was created by Guido van Rossum.", 0, "doc-1")]
    entities, relations = await extractor.extract(chunks)
    print(entities, relations)


asyncio.run(main())
```

### TwoStageArtifactsExtractorLLM

Two-stage LLM extractor.

- Purpose: extract entities first, then extract relations constrained by the entity list.
- Optional validation: `do_entity_validation`, `do_relation_validation`.
- Best for: reducing unresolved relation endpoints.

```python
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI
from ragu.triplet import TwoStageArtifactsExtractorLLM

client = CachedAsyncOpenAI(base_url="https://api.openai.com/v1", api_key="dummy-api-token")
llm = LLMOpenAI(client=client, model_name="gpt-4o-mini")
extractor = TwoStageArtifactsExtractorLLM(
    llm,
    do_entity_validation=True,
    do_relation_validation=True,
)

print(extractor.get_prompt("entity_extraction").pydantic_model)
```

### RaguLmArtifactExtractor

Extractor adapter for RAGU-LM style artifact extraction.

- Purpose: uses chunk context and RAGU-LM prompts to produce graph artifacts.

```python
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI
from ragu.triplet import RaguLmArtifactExtractor

client = CachedAsyncOpenAI(base_url="https://api.openai.com/v1", api_key="dummy-api-token")
llm = LLMOpenAI(client=client, model_name="ragu-lm")
extractor = RaguLmArtifactExtractor(llm, temperature=0.0, top_p=0.95)

print(extractor.temperature)
```

### NEREL Types

`ragu.triplet.types` defines default Russian NEREL-oriented entity and relation type lists used as prompt hints.

```python
from ragu.triplet.types import NEREL_ENTITY_TYPES, NEREL_RELATION_TYPES

print(NEREL_ENTITY_TYPES[:3])
print(NEREL_RELATION_TYPES[:3])
```

## Data Flow

Input: `list[Chunk]`.

Output:

- `list[Entity]` with `source_chunk_id=[chunk.id]`
- `list[Relation]` with endpoint IDs resolved against entities extracted from the same chunk

Used by:

- `ragu.graph.InMemoryGraphBuilder`
- `ragu.graph.KnowledgeGraph.build_from_docs`

## Usage Examples

### Example 1 - Minimal usage

```python
import asyncio

from ragu.chunker.types import Chunk
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI
from ragu.triplet import ArtifactsExtractorLLM


async def main():
    client = CachedAsyncOpenAI(
        base_url="https://api.openai.com/v1",
        api_key="dummy-api-token",
    )
    llm = LLMOpenAI(client=client, model_name="gpt-4o-mini")
    extractor = ArtifactsExtractorLLM(llm, entity_types=["Language"], relation_types=[])
    chunks = [Chunk(content="Python is a programming language.", chunk_order_idx=0, doc_id="doc-1")]
    entities, relations = await extractor.extract(chunks)
    print(entities[0].entity_name, relations)


asyncio.run(main())
```

### Example 2 - Pipeline usage

```python
import asyncio

from ragu import BuilderArguments, KnowledgeGraph, SimpleChunker
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI
from ragu.triplet import TwoStageArtifactsExtractorLLM


async def main():
    client = CachedAsyncOpenAI(
        base_url="https://api.openai.com/v1",
        api_key="dummy-api-token",
    )
    llm = LLMOpenAI(client=client, model_name="gpt-4o-mini")
    embedder = EmbedderOpenAI(
        client=client,
        model_name="text-embedding-3-small",
        dim=1536,
    )
    graph = KnowledgeGraph(
        llm=llm,
        embedder=embedder,
        chunker=SimpleChunker(max_chunk_size=1000, overlap=100),
        artifact_extractor=TwoStageArtifactsExtractorLLM(llm),
        builder_settings=BuilderArguments(make_community_summary=False),
    )
    await graph.build_from_docs(["Python is a programming language."])
    print(await graph.index.graph_backend.get_all_nodes())


asyncio.run(main())
```

## Integration Points

- LLMs: extractors call `llm.batch_chat_completion` with prompt-rendered OpenAI messages.
- Prompt layer: extractors inherit `RaguGenerativeModule` and use `RAGUInstruction`.
- Graph layer: outputs are consumed by `EntitySummarizer`, `RelationSummarizer`, additional builder modules, and `Index`.
- Storage: artifact IDs become graph node/edge IDs and vector record IDs.

## Configuration

`ArtifactsExtractorLLM`:

- `do_validation=False`
- `language=Settings.language`
- `entity_types=NEREL_ENTITY_TYPES`
- `relation_types=NEREL_RELATION_TYPES`

`TwoStageArtifactsExtractorLLM`:

- `do_entity_validation`: disabled unless truthy.
- `do_relation_validation`: disabled unless truthy.
- prompt type hints can be overridden with `entity_types` and `relation_types`.

## Dependencies

Internal:

- `ragu.chunker.types.Chunk`
- `ragu.common.prompts`
- `ragu.graph.types`
- `ragu.models.llm`

External:

- `pydantic`
- `typing_extensions`

## Notes / Pitfalls

- Relations are skipped when their source or target entity name cannot be resolved in the same chunk's extracted entity list.
- Entity IDs are generated from entity name and type, so repeated mentions of the same entity merge later in `KnowledgeGraph`.
- The single-pass extractor may produce relation endpoints that do not exactly match extracted entity names; the two-stage extractor is stricter.
- The `multi_stage_artifacts_extractor.py` file previously referenced in this package has been removed. The two-stage prompts for entity/relation extraction and validation live in `ragu/triplet/prompts.py` and are registered as `entity_extraction`, `entity_validation`, `relation_extraction`, and `relation_validation`.
