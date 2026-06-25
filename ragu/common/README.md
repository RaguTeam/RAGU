# Module: ragu.common

## Role in RAGU Pipeline

`ragu.common` provides cross-cutting infrastructure for the whole GraphRAG pipeline: global settings, prompt rendering, logging, batching, and cache helpers. It is not a pipeline stage by itself, but every stage depends on it.

Pipeline position:

```text
settings + prompts + utilities
  -> chunking / extraction / graph building / storage / retrieval / generation
```

## Overview

The module keeps shared behavior out of the domain packages. It centralizes the default storage directory, default filenames, runtime environment loading, prompt templates, and the base class used by LLM-driven modules.

## Key Components

### Settings

Singleton instance of `GlobalSettings`.

- Purpose: stores process-wide defaults.
- Important fields: `language`, `storage_folder`, tokenizer backends/names,
  token limits (`embedder_token_limit`, `llm_context_token_limit`), cache
  paths (`cache_path`, `debug_errors_path`).
- Tokenizer backend fields (`tokenizer_embedder_backend`,
  `tokenizer_llm_backend`) are typed as `Literal["tiktoken", "local"]`; the
  same constraint applies to the `tokenizer_backend` parameter of
  `EmbedderOpenAI`.
- Used by: storage initialization, prompts, builders, search engines, sparse embedders.
- Serialization: `Settings.save(path)` / `Settings.load(path)` persist and
  restore the user-configurable fields as JSON. Serialization is **never**
  invoked automatically (there are no constructor/destructor hooks): you decide
  when to persist and reload.
  - Serialized fields: `language`, `tokenizer_embedder_backend`,
    `tokenizer_llm_backend`, `tokenizer_embedder_name`, `tokenizer_llm_name`,
    `embedder_token_limit`, `llm_context_token_limit`.
  - **Not** serialized: `storage_folder` (it contains a runtime timestamp by
    default, and restoring it silently would redirect subsequent writes into a
    stale directory — manage it explicitly), `cache_path` and
    `debug_errors_path` (local, machine-specific paths), and internal state
    (singleton handle, current timestamp).
  - `load` validates every value against the declared type hints and raises
    `ValueError` on a mismatch (e.g. an unknown tokenizer backend or a
    non-positive token limit). Unknown keys are reported via a warning and
    ignored, so files produced by a newer RAGU version remain loadable.
- Token-limit defaults: `embedder_token_limit` is used by `EmbedderOpenAI`
  (embedding input truncation); `llm_context_token_limit` is used by search
  engines (LLM context truncation during indexing/answering).
- Cache paths: `cache_path` and `debug_errors_path` default to `None`
  (disabled). When a `CachedAsyncOpenAI` is constructed without an explicit
  `cache` / `debug_errors_storage`, these Settings values are used. They are
  **deliberately independent of `storage_folder`**: the cache must be a stable,
  long-lived path that survives across runs, whereas `storage_folder` is
  per-run. Beware of stale hits — bump `cache_prefix` or clear the directory
  when changing model, temperature, or provider behavior.

```python
from ragu.common.global_parameters import Settings

Settings.language = "english"
Settings.storage_folder = "./ragu_working_dir/demo"
Settings.init_storage_folder()

print(Settings.storage_folder)

# Persist / restore the configuration as a JSON artifact.
Settings.save("./demo/ragu_settings.json")
Settings.load("./demo/ragu_settings.json")
```

### Env

Pydantic settings model.

- Purpose: load model API configuration from environment variables or `.env`.
- Important fields: `llm_model_name`, `llm_base_url`, `llm_api_key`, optional embedder and reranker fields.

```python
from ragu.common.env import Env

env = Env(
    llm_model_name="gpt-4o-mini",
    llm_base_url="https://api.openai.com/v1",
    llm_api_key="dummy-api-token",
)

print(env.llm_model_name)
```

### RaguGenerativeModule

Base class for modules that own prompts.

- Purpose: load default prompts by name or accept custom `RAGUInstruction` objects.
- Important methods: `get_prompt`, `get_prompts`, `update_prompt`.
- Used by: extractors, summarizers, and search engines.

```python
from ragu.common.base import RaguGenerativeModule
from ragu.common.prompts.messages import ChatMessages, UserMessage
from ragu.common.prompts.prompt_storage import RAGUInstruction

module = RaguGenerativeModule(prompts=["naive_search"])
module.update_prompt(
    "naive_search",
    RAGUInstruction(
        messages=ChatMessages.from_messages([
            UserMessage(content="Answer in {{ language }} using this context:\n{{ context }}\n\nQuery: {{ query }}")
        ]),
        pydantic_model=str,
        description="Custom concise answer prompt.",
    ),
)

print(module.get_prompt("naive_search").description)
```

### ChatMessages and Message Types

Prompt-message abstraction from `ragu.common.prompts.messages`.

- Purpose: represent system/user/assistant messages and convert them to OpenAI chat payloads.
- Important classes: `SystemMessage`, `UserMessage`, `AIMessage`, `ChatMessages`.

```python
from ragu.common.prompts.messages import ChatMessages, SystemMessage, UserMessage

messages = ChatMessages.from_messages([
    SystemMessage(content="Answer briefly."),
    UserMessage(content="What is RAGU?"),
])

print(messages.to_openai())
```

### render

Jinja2 renderer for prompt messages.

- Purpose: render one or many conversations from scalar and batch parameters.
- Important behavior: list/tuple parameters define batch size; all batch parameters must have the same length.

```python
from ragu.common.prompts.messages import ChatMessages, UserMessage, render

template = ChatMessages.from_messages([
    UserMessage(content="Question: {{ query }}")
])

rendered = render(template, query=["What is RAGU?", "What is GraphRAG?"])
print([conversation.to_str() for conversation in rendered])
```

### BatchGenerator

Small batching helper used by rerankers and utility code.

```python
from ragu.common.batch_generator import BatchGenerator

generator = BatchGenerator([1, 2, 3, 4, 5], batch_size=2)
print(list(generator.get_batches()))
```

### get_cache

Disk cache helper that returns a mutable mapping backed by `diskcache`.

```python
from ragu.common.cache import get_cache

cache = get_cache("./ragu_working_dir/cache")
cache["key"] = {"value": 1}
print(cache["key"])
```

## Data Flow

Input: runtime configuration, prompt names, prompt parameters.

Output: storage paths, rendered `ChatMessages`, OpenAI-compatible message lists, cache mappings.

Used by:

- `ragu.triplet` extraction prompts
- `ragu.graph` summarization prompts and storage defaults
- `ragu.search_engine` answer-generation prompts
- `ragu.models` caching and API wrappers

## Usage Examples

### Example 1 - Minimal usage

```python
from ragu.common.global_parameters import Settings

Settings.language = "english"
Settings.storage_folder = "./ragu_working_dir/example"
Settings.init_storage_folder()

print(Settings.storage_folder)
```

### Example 2 - Pipeline usage

```python
from ragu.common.prompts.messages import ChatMessages, SystemMessage, UserMessage, render

template = ChatMessages.from_messages([
    SystemMessage(content="Answer in {{ language }}."),
    UserMessage(content="Question: {{ query }}\nContext: {{ context }}"),
])

rendered = render(
    template,
    language="english",
    query=["What is RAGU?", "What is local search?"],
    context=["GraphRAG system", "Entity-neighborhood retrieval"],
)

openai_messages = [conversation.to_openai() for conversation in rendered]
print(openai_messages[0])
```

### Example 3 - Change an instruction in a generative module

```python
from ragu.common.base import RaguGenerativeModule
from ragu.common.prompts.messages import ChatMessages, UserMessage, render
from ragu.common.prompts.prompt_storage import RAGUInstruction

module = RaguGenerativeModule(prompts=["local_search"])
module.update_prompt(
    "local_search",
    RAGUInstruction(
        messages=ChatMessages.from_messages([
            UserMessage(
                content=(
                    "Use only the context below. "
                    "Answer in {{ language }}.\n\n"
                    "Context:\n{{ context }}\n\n"
                    "Question: {{ query }}"
                )
            )
        ]),
        pydantic_model=str,
        description="Strict context-only local search prompt.",
    ),
)

instruction = module.get_prompt("local_search")
rendered = render(
    instruction.messages,
    language="english",
    context="Python is a programming language.",
    query="What is Python?",
)[0]

print(rendered.to_openai())
```

## Integration Points

- LLMs: `ChatMessages.to_openai()` produces payloads accepted by `LLM.chat_completion`.
- Extraction and retrieval: `RaguGenerativeModule` loads named prompts from `DEFAULT_PROMPT_TEMPLATES`.
- Storage: `Settings.storage_folder` and `DEFAULT_FILENAMES` define default locations for KV, vector, and graph files.
- Configuration: `Env.from_env()` loads OpenAI-compatible model settings for application entrypoints.

## Configuration

Environment variables consumed by `Env`:

- `llm_model_name`, `llm_base_url`, `llm_api_key`
- `embedder_base_url`, `embedder_api_key`, `embedder_model_name`
- `reranker_base_url`, `reranker_api_key`, `reranker_model_name`

Global defaults:

- `Settings.language = "english"`
- `Settings.storage_folder` defaults to `./ragu_working_dir/<timestamp>`.

## Dependencies

Internal:

- `ragu.common.prompts`
- `ragu.common.logger`

External:

- `pydantic-settings`
- `jinja2`
- `diskcache`
- `loguru`
- OpenAI SDK message types

## Notes / Pitfalls

- `Settings` is global process state. Set `Settings.storage_folder` before constructing `KnowledgeGraph` or `Index`.
- The `tokenizer_*_name` fields drive only token counting for truncation in
  `EmbedderOpenAI` and `BaseEngine`; they are **not** derived from `Env`.
  When you switch the LLM via `Env.llm_model_name`, update
  `Settings.tokenizer_llm_name` accordingly (and `Settings.tokenizer_embedder_name`
  when you swap the embedder). Otherwise truncation is computed against the
  wrong tokenizer's token counts.
- Jinja rendering uses `StrictUndefined`; missing template variables raise errors.
- `render()` treats any list or tuple parameter as batched input.
- Default prompt names must exist in `DEFAULT_PROMPT_TEMPLATES`, otherwise `get_prompt()` can return `None` at construction time and fail later.
