# Module: ragu.utils.testing

## Role in RAGU Pipeline

`ragu.utils.testing` provides test doubles for RAGU's external dependencies. It
is not part of the indexing or retrieval pipeline; it exists to make tests of
the model layer (`LLMOpenAI`, `EmbedderOpenAI`, structured-output decoding,
retries, rate limiting) deterministic and free of network access.

Pipeline position:

```text
test -> OpenAIMockServer -> CachedAsyncOpenAI -> LLM/Embedder under test
```

## Overview

The package ships `OpenAIMockServer`, a thread-backed `http.server` that
returns minimal valid OpenAI-compatible responses. It is the basis of the model
tests in `tests/` and of any user-facing test suite that exercises RAGU without
a real provider.

## Key Components

### OpenAIMockServer

Thread-backed mock HTTP server from `ragu.utils.testing.openai_mock_server`.

- Purpose: respond to `POST /v1/chat/completions`, `POST /v1/embeddings`, and
  `POST /v1/score` with minimal valid payloads.
- Important methods: `start()`, `stop()`, and the `base_url` property.
- Fault injection: supports configurable response delays, HTTP error codes,
  rate-limit simulation (`min_delay`), and intentionally malformed bodies.
- Used by: `tests/models/`, `tests/triplet/`, and other tests that need a
  deterministic LLM/embedder endpoint.

```python
from ragu.utils.testing.openai_mock_server import OpenAIMockServer

server = OpenAIMockServer("127.0.0.1", 0)  # 0 = pick a free port
server.start()

print(server.base_url)   # http://127.0.0.1:<port>/v1/ — pass to CachedAsyncOpenAI(base_url=...)

server.stop()
```

Wire it into the client exactly like a real endpoint:

```python
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI

client = CachedAsyncOpenAI(base_url=server.base_url, api_key="mock")
llm = LLMOpenAI(client=client, model_name="mock-model")
```

## Data Flow

Input: HTTP requests issued by `CachedAsyncOpenAI`.

Output: minimal valid OpenAI-compatible JSON responses, with optional
fault-injection behavior controlled by the server instance.

Used by:

- `tests/models/`
- `tests/triplet/`
- any user test suite that exercises RAGU model wrappers

## Integration Points

- `CachedAsyncOpenAI` accepts `server.base_url` as its `base_url`.
- Structured-output decoding paths (`pydantic_model`) can be exercised by
  configuring the mock response body.
- Retry and rate-limit behavior can be tested via the server's fault injection.

## Configuration

- `OpenAIMockServer(host, port=0)`: `port=0` binds an OS-chosen free port,
  exposed afterwards via `server.base_url` / `server.server_address`.
- `default_delay=(min, max)`: per-request random sleep range.
- `min_delay`: minimum seconds between accepted requests; faster requests get
  HTTP 429 (rate-limit simulation).

## Dependencies

Internal:

- `ragu.models.openai.CachedAsyncOpenAI` (consumed in tests)

External:

- Python standard library (`http.server`, `threading`)

## Notes / Pitfalls

- Always call `server.stop()` (e.g. in a `finally` block or fixture teardown) to
  join the background daemon thread and free the port.
- The mock returns minimal payloads; it does not validate semantic correctness
  of model outputs. Tests that depend on specific content must configure the
  response body explicitly.
- These utilities are for local tests, not production serving.
