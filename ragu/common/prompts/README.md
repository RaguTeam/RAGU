# Module: ragu.common.prompts

## Role in RAGU Pipeline

`ragu.common.prompts` is the prompt layer of RAGU. It defines the conversation
abstraction used by every LLM call, the registry of built-in instructions, the
Jinja2 rendering helpers, and the in-context learning (few-shot) machinery.

Pipeline position:

```text
RAGUInstruction + parameters -> render -> ChatMessages -> LLM
```

## Overview

The module keeps prompt definition, prompt rendering, and few-shot example
selection out of the domain modules. An instruction is a frozen
`RAGUInstruction` dataclass that binds a `ChatMessages` template, an optional
Pydantic schema for structured output, an optional description, and an optional
few-shot formatter. Instructions are looked up by name through
`DEFAULT_PROMPT_TEMPLATES` and customized per module instance through
`RaguGenerativeModule.update_prompt`.

## Key Components

### RAGUInstruction

Frozen dataclass from `ragu.common.prompts.prompt_storage`.

- Purpose: bind a conversation template together with its output schema and
  optional metadata.
- Fields:
  - `messages: ChatMessages` — the Jinja2-templated conversation.
  - `pydantic_model: Type[BaseModel] | Type[str]` — schema used for structured
    decoding (`str` means free-form text).
  - `description: str | None` — short human-readable summary (optional).
  - `few_shot_formatter: FewShotFormatter | None` — formats each few-shot
    example into a `(UserMessage, AIMessage)` pair when ICL is enabled.
- Customization rule: never edit `DEFAULT_PROMPT_TEMPLATES` directly. Override
  a name on a module instance via `RaguGenerativeModule.update_prompt`.

```python
from ragu.common.prompts.messages import ChatMessages, UserMessage
from ragu.common.prompts.prompt_storage import RAGUInstruction

instruction = RAGUInstruction(
    messages=ChatMessages.from_messages([
        UserMessage(content="Answer in {{ language }}.\nQuery: {{ query }}"),
    ]),
    pydantic_model=str,
    description="Minimal answer prompt.",
)

print(instruction.description)
```

### ChatMessages and Message Types

Conversation abstraction from `ragu.common.prompts.messages`.

- Purpose: represent a system/user/assistant conversation and convert it to
  OpenAI-compatible payloads.
- Important classes: `BaseMessage`, `SystemMessage`, `UserMessage`,
  `AIMessage`, `ChatMessages`.
- Important methods: `ChatMessages.from_messages(...)`, `.to_openai()`,
  `.to_str()`.

```python
from ragu.common.prompts.messages import ChatMessages, SystemMessage, UserMessage

messages = ChatMessages.from_messages([
    SystemMessage(content="Answer briefly."),
    UserMessage(content="What is RAGU?"),
])

print(messages.to_openai())
```

### render and render_with_few_shots

Jinja2 renderers from `ragu.common.prompts.messages`.

- Purpose: render one or many conversations from scalar and batch parameters.
- `render(template, **params)`: list/tuple parameters define batch size; all
  batch parameters must share the same length. Uses `StrictUndefined`, so
  missing template variables raise.
- `render_with_few_shots(template, examples_list, few_shot_formatter, **params)`:
  same as `render`, but inserts `(user, ai)` few-shot pairs before the last
  message of each rendered conversation. When `few_shot_formatter` is `None` or
  no examples are supplied, behaves like `render`.

```python
from ragu.common.prompts.messages import ChatMessages, UserMessage, render

template = ChatMessages.from_messages([
    UserMessage(content="Question: {{ query }}")
])

rendered = render(template, query=["What is RAGU?", "What is GraphRAG?"])
print([conversation.to_str() for conversation in rendered])
```

### DEFAULT_PROMPT_TEMPLATES

Registry mapping instruction names to `RAGUInstruction` instances. There is no
auto-discovery: every built-in prompt is registered in
`ragu/common/prompts/prompt_storage.py`. Modules load their prompts by name
through `RaguGenerativeModule`.

| Name | Pydantic model | Used by |
|------|----------------|---------|
| `artifact_extraction` | `ArtifactsModel` | `ArtifactsExtractorLLM` |
| `artifact_validation` | `ArtifactsModel` | `ArtifactsExtractorLLM` |
| `community_report` | `CommunityReportModel` | `CommunitySummarizer` |
| `entity_summarizer` | `EntityDescriptionModel` | `EntitySummarizer` |
| `relation_summarizer` | `RelationDescriptionModel` | `RelationSummarizer` |
| `global_search_context` | `GlobalSearchContextModel` | `GlobalSearchEngine` |
| `global_search` | `str` | `GlobalSearchEngine` |
| `local_search` | `str` | `LocalSearchEngine` |
| `naive_search` | `str` | `NaiveSearchEngine` |
| `mix_search` | `str` | `MixSearchEngine` |
| `mix_search_context` | `str` | `MixSearchEngine` |
| `cluster_summarize` | `ClusterSummarizationModel` | clustering pipeline |
| `ragu_lm_entity_extraction` | `str` | `RaguLmArtifactExtractor` |
| `ragu_lm_entity_normalization` | `str` | `RaguLmArtifactExtractor` |
| `ragu_lm_entity_description` | `str` | `RaguLmArtifactExtractor` |
| `ragu_lm_relation_description` | `str` | `RaguLmArtifactExtractor` |
| `query_decomposition` | `QueryPlan` | `QueryPlanEngine` |
| `query_rewrite` | `RewriteQuery` | `QueryPlanEngine` |

```python
from ragu.common.prompts import DEFAULT_PROMPT_TEMPLATES

print(sorted(DEFAULT_PROMPT_TEMPLATES))
print(DEFAULT_PROMPT_TEMPLATES["local_search"].pydantic_model)
```

### ICLConfig

Frozen dataclass from `ragu.common.prompts.icl_config`.

- Purpose: configure few-shot example selection for extractors.
- Fields: `enabled`, `num_examples`, `examples_base_path`, `selection_strategy`,
  `low_match_warning_threshold`.
- Strategies:
  - `"semantic"`: cosine similarity on dense embeddings (default). Requires an
    `Embedder`.
  - `"bm25"`: lexical matching via FastEmbed BM25. No embedder needed.
  - `"hybrid"`: Reciprocal Rank Fusion of semantic and BM25. Requires an
    `Embedder`.
  - `"random"`: uniform random sampling. Baseline; needs no embedder.

```python
from ragu.common.prompts import ICLConfig

icl_config = ICLConfig(
    enabled=True,
    num_examples=2,
    selection_strategy="hybrid",
)
```

### InContextLearningManager

ICL engine from `ragu.common.prompts.icl_manager`.

- Purpose: load JSON example files, build embeddings/BM25 index, and select
  examples per query.
- Built-in examples ship in `ragu/common/prompts/icl_examples/`
  (`artifact_extraction_examples.json`,
  `artifact_validation_examples.json`,
  `entity_extraction_examples.json`,
  `entity_validation_examples.json`,
  `relation_extraction_examples.json`,
  `relation_validation_examples.json`).
- Override the directory with `ICLConfig.examples_base_path` to use custom
  examples.

### FewShotFormatter

Callable type and ready formatters from `ragu.common.prompts.few_shot`.

- Signature: `Callable[[dict[str, Any]], tuple[UserMessage, AIMessage]]`.
- Built-in formatters: `format_artifact_extraction_example`,
  `format_artifact_validation_example`,
  `format_entity_extraction_example`,
  `format_entity_validation_example`,
  `format_relation_extraction_example`,
  `format_relation_validation_example`.
- Bound to an instruction via `RAGUInstruction.few_shot_formatter`.

## Data Flow

Input: instruction name plus scalar or batched render parameters.

Output: rendered `ChatMessages` ready for `llm.chat_completion` /
`llm.batch_chat_completion`, optionally enriched with few-shot pairs.

Used by:

- `ragu.triplet` extraction and validation prompts
- `ragu.graph` entity/relation/community summarization prompts
- `ragu.search_engine` answer-generation prompts
- `ragu.search_engine.QueryPlanEngine` decomposition/rewrite prompts

## Usage Examples

### Example 1 - Inspect and override a module prompt

```python
from ragu import LocalSearchEngine

search_engine = LocalSearchEngine(
    llm=llm,
    knowledge_graph=knowledge_graph,
    embedder=embedder,
)

all_prompts = search_engine.get_prompts()
local_prompt = search_engine.get_prompt("local_search")
print(local_prompt.messages.to_str())
print(local_prompt.pydantic_model)
```

```python
from textwrap import dedent

from ragu.common.prompts.messages import ChatMessages, SystemMessage, UserMessage
from ragu.common.prompts.prompt_storage import RAGUInstruction

search_engine.update_prompt(
    "local_search",
    RAGUInstruction(
        messages=ChatMessages.from_messages([
            SystemMessage(content="Answer using only the supplied graph context."),
            UserMessage(content=dedent(
                """
                Query: {{ query }}
                Context: {{ context }}
                Language: {{ language }}
                """
            )),
        ]),
        pydantic_model=str,
        description="Custom local-search instruction.",
    ),
)
```

### Example 2 - Enable ICL on an extractor

```python
from ragu.common.prompts import ICLConfig
from ragu.triplet import ArtifactsExtractorLLM

icl_config = ICLConfig(
    enabled=True,
    num_examples=2,
    selection_strategy="semantic",
)

extractor = ArtifactsExtractorLLM(
    llm=llm,
    embedder=embedder,
    icl_config=icl_config,
)
```

## Integration Points

- LLMs: `ChatMessages.to_openai()` produces payloads accepted by
  `LLM.chat_completion` / `LLM.batch_chat_completion`.
- Generative modules: every `RaguGenerativeModule` subclass loads its prompts
  by name from `DEFAULT_PROMPT_TEMPLATES` at construction time.
- Extractors: `ArtifactsExtractorLLM` and `TwoStageArtifactsExtractorLLM`
  register the ICL manager for the extraction/validation tasks.
- Storage: built-in example JSON files are shipped as package data
  (`pyproject.toml`:
  `"ragu" = ["common/prompts/icl_examples/*.json"]`).

## Configuration

ICL selection strategy trade-offs:

- `"semantic"`: best for paraphrased / synonym-rich queries; requires an
  embedder and pays the embedding cost.
- `"bm25"`: fastest, no embedder; good for terminology overlap.
- `"hybrid"`: combines both via Reciprocal Rank Fusion; requires an embedder.
- `"random"`: baseline only.

## Dependencies

Internal:

- `ragu.common.logger`
- `ragu.models.embedder` (for `"semantic"` / `"hybrid"` ICL strategies)
- `ragu.models.sparse_embedder.BM25` (for `"bm25"` / `"hybrid"` strategies)

External:

- `jinja2`
- `pydantic`
- OpenAI SDK message types

## Notes / Pitfalls

- Jinja rendering uses `StrictUndefined`; missing template variables raise.
- `render()` treats any `list` or `tuple` parameter as batched input, and all
  batched parameters must share the same length.
- Do not mutate `DEFAULT_PROMPT_TEMPLATES` to "fix" one call site. Use
  `module.update_prompt(name, instruction)` so the change is scoped to a
  single module instance.
- The `pydantic_model` bound to an instruction must match the schema expected
  by the consuming module; changing it can break structured-output decoding.
