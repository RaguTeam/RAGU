# Module: ragu.triplet

## Role in RAGU Pipeline

`ragu.triplet` is the extraction layer. It converts chunks into graph artifacts: entities and relations, restricted to an ontology. These artifacts are then summarized, clustered, and stored by `ragu.graph`.

Pipeline position:

```text
List[Chunk] -> artifact extractor -> List[Entity], List[Relation] -> graph builder
```

## Overview

The module exists to isolate LLM prompting and structured artifact conversion from graph storage. Extractors use prompts and Pydantic schemas to get structured output, then convert that output into `Entity` and `Relation` dataclasses with stable IDs and source chunk references.

Every LLM extractor takes a single `ontology` argument. That one object supplies the type list injected into the prompt **and** the rules enforced on what comes back, so the two cannot drift apart. `validation` selects what to do about violations.

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
- Important parameters: `llm`, `language`, `ontology`, `validation`, `show_type_signatures`.

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

### Ontology

`Ontology` carries the vocabulary and, optionally, the rules: `domain`/`range` per predicate, a type hierarchy, symmetric and inverse predicates, `aliases` and `inverse_aliases` per type, and `retype_when` rules that rewrite a predicate for specific endpoint types. A flat type list is the degenerate case where no predicate declares constraints.

```python
from ragu.triplet import Ontology

nerel = Ontology.builtin("nerel")          # 29 entity types, 49 predicates, with rules
flat = Ontology.from_type_lists(["PERSON (a human)"], ["KNOWS"])   # vocabulary only
custom = Ontology.from_yaml("legal.yaml")               # may `extends: nerel`

print(nerel.allows("WORKPLACE", "PERSON", "ORGANIZATION"), nerel.ancestors("CITY"))
```

`ragu/triplet/ontology/builtin/nerel.yaml` is the single source of truth for NEREL; the legacy `NEREL_ENTITY_TYPES` / `NEREL_RELATION_TYPES` lists in `ragu.triplet.types` are generated from it.

### OntologyValidator

Applied by every LLM extractor between the model output and `Entity`/`Relation` construction — before construction, because `Entity.id` hashes the entity type. Checks: type outside the vocabulary, `domain`/`range` violation, self-loop, endpoint missing from the chunk.

```python
from ragu.triplet import Ontology, OntologyValidator, ValidationPolicies

validator = OntologyValidator(Ontology.builtin("nerel"), ValidationPolicies.strict())
entities, relations, report = validator.validate(entities, relations)
print(report.summary())
```

`ValidationPolicies` chooses what happens per check: `COERCE` (default) repairs, `DROP` discards, `KEEP` only counts, `RAISE` aborts. Presets: `permissive()`, `strict()`, `failing()`.

A `COERCE`d violation is repaired in one of these ways:

- an unknown type resolves through `aliases` (`FILM` → `WORK_OF_ART`), or through `inverse_aliases`, which also swaps the endpoints (`HAS_PART` → `PART_OF`);
- an unknown type with no alias is matched fuzzily against the vocabulary (`ORGANISATION` → `ORGANIZATION`);
- a `domain`/`range` violation is repaired by a `retype_when` rule (`PERSON -LOCATED_IN-> CITY` becomes `PLACE_RESIDES_IN`; a rule with `swap: true` also exchanges the endpoints, turning `WORK_OF_ART -AGENT-> PERSON` into `PERSON -PRODUCES-> WORK_OF_ART`), or by swapping the endpoints when the reversed triple is valid.

Every repair only applies if the result actually satisfies the ontology. `inverse_aliases` exist because a plain alias is direction-blind and the swap repair cannot detect a flipped relation when `domain` equals `range` (`CHILD_OF` / `PARENT_OF`) or the predicate is unconstrained (`HAS_PART` / `PART_OF`).

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
from ragu.triplet import ArtifactsExtractorLLM, Ontology


async def main():
    client = CachedAsyncOpenAI(
        base_url="https://api.openai.com/v1",
        api_key="dummy-api-token",
    )
    llm = LLMOpenAI(client=client, model_name="gpt-4o-mini")
    extractor = ArtifactsExtractorLLM(
        llm,
        ontology=Ontology.from_type_lists(["LANGUAGE (a programming or natural language)"], []),
    )
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

Both LLM extractors:

- `ontology="nerel"` — an `Ontology`, the name of a built-in one, or `None` to let the model invent its own labels and skip all checks.
- `validation=ValidationPolicies()` — coerce unknown types and `domain`/`range` violations, drop self-loops and dangling endpoints.
- `show_type_signatures=False` — render each predicate as `WORKPLACE [PERSON -> ORGANIZATION|FACILITY] (...)` and add a short legend explaining the notation and the endpoint order. Off by default because it changes the prompt, but worth turning on: without it the model cannot know which endpoint types a predicate accepts, and most discarded relations are exactly that mistake. The list and the legend live in the system message, so on a vLLM backend with prefix caching they are prefilled once for the whole batch.
- `language=Settings.language`

`ArtifactsExtractorLLM`:

- `do_validation=False` — an extra LLM pass over the extracted artifacts.

`TwoStageArtifactsExtractorLLM`:

- `do_entity_validation` / `do_relation_validation`: disabled unless truthy.
- `prune_relation_types=False` — offer each chunk only the predicates admissible between the entity types found in it. Possible here because the entity stage runs first; typically cuts the predicate list several-fold. Note the trade-off on vLLM: this makes the system message unique per chunk and therefore defeats prefix caching, so the shared full list can end up cheaper for batches larger than a handful of chunks.

`RaguLmArtifactExtractor` takes no ontology: it labels every artifact `UNKNOWN`.

## Dependencies

Internal:

- `ragu.chunker.types.Chunk`
- `ragu.common.prompts`
- `ragu.graph.types`
- `ragu.models.llm`

External:

- `pydantic`
- `pyyaml` — ontology documents
- `typing_extensions`

## Notes / Pitfalls

- Relations are skipped when their source or target entity name cannot be resolved in the same chunk's extracted entity list.
- Entity IDs are generated from entity name and type, so repeated mentions of the same entity merge later in `KnowledgeGraph`. This is also why ontology validation runs on the Pydantic stage models rather than on `Entity`: a type corrected after construction would leave a stale ID.
- Entity names and relation endpoints are stripped by `EntityModel` / `RelationModel`, because `Entity.id` hashes the name: `" eurozone"` and `"eurozone"` would otherwise be two nodes.
- Ontology validation discards artifacts. When a graph comes out sparser than expected, look for the `Ontology validation (...)` warning: besides counts per check it names the top signatures — what was dropped (`WORKPLACE (PERSON -> CITY)=6`) and what each repair turned into (`WORKPLACES -> WORKPLACE=4`, `AGENT (EVENT -> PERSON) swapped=2`). The full, untruncated list goes to the `DEBUG` log, or call `ValidationReport.breakdown()` when driving the validator yourself.
- In the two-stage extractor entities are validated *before* the relation stage, so the relation prompt only ever sees entities that survived.
- The single-pass extractor may produce relation endpoints that do not exactly match extracted entity names; the two-stage extractor is stricter.
