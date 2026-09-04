# RAGU HTTP service

---
1. [What it is](#what-it-is)
2. [Running the service](#running-the-service)
3. [Configuration](#configuration)
4. [Search endpoints](#search-endpoints)
5. [Health and readiness](#health-and-readiness)
6. [Errors](#errors)
7. [Design decisions](#design-decisions)
8. [Logging](#logging)
9. [Current limits](#current-limits)

---

## What it is

`ragu.api` is a FastAPI service that puts a prebuilt knowledge graph behind
HTTP. It ships inside the package, so an installed RAGU can serve a graph
without a checkout, and clients in any language can query it instead of
importing RAGU as a Python library.

The service **serves** a graph; it never builds one. Build the graph first —
`KnowledgeGraph.build_from_docs`, or one of the scripts in `examples/` — then
point the service at that storage folder.

It owns one graph per process and exposes one route per search mode, so a
gateway can apply a different timeout and rate limit to each. That matters:
global search rates every community summary against the query with its own LLM
call (N+1 calls for N communities), while local and naive issue one generation
each.

Install it with the `api` extra:

```bash
uv pip install -e ".[api]"
```

## Running the service

```bash
# Canned answers, no graph and no LLM required — for developing clients
RAGU_API_BACKEND=stub python -m ragu.api --port 8020

# Real graph
RAGU_API_STORAGE_FOLDER=./ragu_working_dir \
RAGU_API_LANGUAGE=russian \
python -m ragu.api --port 8020
```

`--host`, `--port`, `--backend` and `--storage-folder` override the
corresponding environment variables on the command line. Interactive API docs
are at `/docs`.

With Docker, `docker compose up -d ragu-api` builds the image from the
repository root and mounts a prebuilt graph at `/data/graph`. The deployment
details that bite — the whitelist `.dockerignore`, baked tiktoken vocabularies,
the healthcheck start period, and write access for uid 1000 — are documented in
[`ragu/api/README.md`](../../ragu/api/README.md).

## Configuration

Service settings are `RAGU_API_*` environment variables, read by
`ServiceSettings` (`ragu/api/config.py`), optionally from a `.env` file.

| Variable | Default | Meaning |
|---|---|---|
| `RAGU_API_BACKEND` | `ragu` | `ragu` loads a real graph, `stub` serves canned answers |
| `RAGU_API_HOST` | `0.0.0.0` | Bind address |
| `RAGU_API_PORT` | `8020` | Bind port |
| `RAGU_API_STORAGE_FOLDER` | `ragu_working_dir` | Folder holding the built graph |
| `RAGU_API_LANGUAGE` | `russian` | Passed to `Settings.language` |
| `RAGU_API_SETTINGS_FILE` | — | `Settings` JSON saved at build time, loaded instead of the defaults |
| `RAGU_API_EMBEDDER_DIM` | — | Embedding dimension; auto-detected with a probe request when unset |
| `RAGU_API_RATE_MIN_DELAY` | — | Minimum delay between LLM calls, seconds |
| `RAGU_API_RATE_MAX_SIMULTANEOUS` | — | Maximum simultaneous LLM calls |
| `RAGU_API_LLM_CACHE` | — | Path to the LLM response cache; unset disables caching |
| `RAGU_API_MAX_TOP_K` | `100` | Ceiling applied to a client-supplied `top_k` / `rerank_top_k` |
| `RAGU_API_MIN_CLUSTER_SIZE_FLOOR` | `1` | Floor applied to global `min_cluster_size` |
| `RAGU_API_STUB_MISSING_CAPABILITIES` | — | Stub only: capabilities to report as missing |

`RAGU_API_STORAGE_FOLDER` must point at one graph directory, not at a folder of
them, and the directory must already exist and be non-empty — the service
refuses to start otherwise rather than creating it.

LLM and embedder credentials come from the same `.env` the rest of RAGU uses
(`ragu.common.env.Env`): `LLM_MODEL_NAME`, `LLM_BASE_URL`, `LLM_API_KEY`, and
optionally `EMBEDDER_BASE_URL`, `EMBEDDER_API_KEY`, `EMBEDDER_MODEL_NAME`.

## Search endpoints

| Route | Body |
|---|---|
| `POST /v1/search/global` | `query`, `params: GlobalSearchParams` |
| `POST /v1/search/local` | `query`, `use_query_plan=true`, `params: LocalParams` |
| `POST /v1/search/naive` | `query`, `use_query_plan=true`, `params: NaiveSearchParams` |

`params` is the engine's own parameter class, embedded in the request model
rather than restated field by field:

```json
{
  "query": "Who wrote the novel?",
  "use_query_plan": true,
  "params": {"top_k": 20, "rerank_top_k": null, "use_summary": true, "use_chunks": true}
}
```

Three consequences of that choice:

- **Defaults are the engine's.** Omitting `params` yields `LocalParams()` /
  `NaiveSearchParams()`, so `use_summary` is `false` and `top_k` is `20`.
  Clients that care should send the values.
- **Bounds are the service's.** The dataclasses carry none, so `top_k` and
  `rerank_top_k` are clamped to `RAGU_API_MAX_TOP_K` and `min_cluster_size` is
  raised to `RAGU_API_MIN_CLUSTER_SIZE_FLOOR`.
- **Unknown fields are rejected, inside `params` as well as outside it.** A
  top-level `top_k` (the shape before `params` existed) and a mistyped
  `params.topk` both answer `400` instead of quietly serving the engine default.

Global search takes no `use_query_plan`: its answer is synthesized from
summaries that already cover the whole corpus, so decomposing the question would
only re-rate the same summaries once per subquery. `used_query_plan` stays in
the response envelope and is always `false` for that mode.

Success response:

```json
{
  "query": "Who wrote the novel 'Quo Vadis'?",
  "mode": "local",
  "used_query_plan": true,
  "answer": "The novel was written by Henryk Sienkiewicz...",
  "sources": [{"id": "chunk-8f3...", "type": "chunk", "content": "...", "score": 0.87}],
  "subqueries": [{"query": "Who wrote the novel?", "answer": "Henryk Sienkiewicz"}]
}
```

`sources` is the retrieval flattened to `{id, type, content, score}`: chunks and
their scores from `NaiveSearchResult`, entities / relations / summaries / chunks
from `LocalSearchResult`, rated insights from `GlobalSearchResult`. A result
type the service does not model degrades to a single source rendered with
`to_text()`.

When a query plan ran, `sources` carries the evidence of **every** subquery,
deduplicated, not only that of the final one — the intermediate answers are
returned in `subqueries`, so their evidence is returned with them.

## Health and readiness

| Route | Status | Use |
|---|---|---|
| `GET /health` | always `200` | Human/debug view; body says whether it is ready |
| `GET /health/live` | always `200` | Liveness probe: the process is up |
| `GET /health/ready` | `200` / `503` + `Retry-After` | Readiness probe: searches can be served |

Loading a large graph takes minutes, so liveness and readiness are separate: an
orchestrator must not restart a service that is still starting up.

```json
{
  "status": "ok",
  "graph_loaded": true,
  "stats": {"entities": 128400, "relations": 170233, "chunks": 9812, "community_summaries": 341},
  "error": null
}
```

`stats` is measured once at startup and is what `graph_loaded` is derived from:
it means "the graph holds something a search mode can read", not "an object was
constructed". When startup failed, `error` carries the reason and the same
reason appears in the `503` body of every search — the diagnostics are not
buried in the log.

## Errors

All errors share one envelope:

```json
{"error": {"code": "CAPABILITY_UNAVAILABLE", "mode": "global",
           "missing_capability": "community_summaries", "message": "..."}}
```

| Status | Code | Situation |
|---|---|---|
| 400 | `INVALID_REQUEST` | empty `query`, bad field type, unknown field |
| 409 | `CAPABILITY_UNAVAILABLE` | the graph cannot serve this mode, or this query found nothing |
| 503 | `SERVICE_NOT_READY` | graph not loaded, or startup failed |
| 500 | `INTERNAL_ERROR` | LLM unavailable, embedder timeout, engine failure |

`409` exists so a client can react on its own: a graph built without community
summaries cannot answer global searches, and one built without entity extraction
cannot answer local ones. `missing_capability` tells the two `409` cases apart:

- **named** (`community_summaries`, `entity_graph`, `vector_index`) — the graph
  has nothing this mode reads. Detected from the startup measurement, so no
  generation call is spent on it.
- **`null`** — the graph does support the mode, but this particular query
  retrieved no evidence. The engines answer confidently from an empty context,
  so an answer built on nothing is reported rather than returned.

`500` responses never quote the underlying exception: engine and LLM-client
errors routinely include the endpoint URL and parts of the request. The detail
goes to the service log.

## Design decisions

**Why the request embeds the engine's parameter class.** `QueryPlanEngine`
forwards `params` untouched to the engine it wraps, so per-request options have
to travel as parameter objects rather than constructor arguments. Reusing those
same classes as the request schema means an engine parameter added upstream is
available over HTTP with no change here — and one renamed upstream changes the
wire contract, which is the price of not restating them.

**Why capabilities are measured, not inferred.** The engines answer from
whatever context they gathered, empty included. Counting the entity, chunk and
community-summary stores once at startup is what lets the service refuse an
unsupported mode before paying for a generation call, and what makes
`graph_loaded` truthful.

**Why conversion dispatches on result types.** The engines are part of this
package and versioned with it, so `ragu/api/mapping.py` dispatches on the
concrete `*SearchResult` classes instead of probing for attributes. A result
type that changes shape then fails loudly instead of producing an empty source
list that the service would report as a missing capability.

**Why the cost knobs are server-side.** One global search costs N+1 LLM calls
for N surviving communities. `RAGU_API_MIN_CLUSTER_SIZE_FLOOR` is the only cap
on that, just as `RAGU_API_MAX_TOP_K` caps retrieval width; both are applied in
the backend base class so every backend enforces them identically.

## Logging

`python -m ragu.api` calls `configure_logging` (`ragu/api/logging_setup.py`),
which installs an intercept handler on the stdlib root and re-emits every
record through loguru. Uvicorn, httpx and the service itself then share the one
sink and one format the engines already use, so a single request can be
followed end to end. `--log-level` sets that sink's level.

Importing `create_app` into another server (gunicorn, an existing uvicorn
setup) leaves that server's logging untouched — call `configure_logging`
yourself if you want the same behaviour.

## Current limits

Known gaps, listed so they are not mistaken for oversights:

- **One graph per process.** `Settings` is a process-wide singleton and
  `Index` reads `Settings.storage_folder` in its constructor, so a process
  serves exactly one graph.
- **No ingestion over HTTP.** The service cannot build or extend a graph;
  `build_from_docs` and the graph CRUD surface of `KnowledgeGraph` are not
  exposed.
- **Search only, one query at a time.** `MixSearchEngine`, retrieval without
  generation (`batch_search`), batched queries (`batch_query`) and streaming
  (`stream_query`) are all supported by the engines and not yet exposed.
- **No authentication, quotas or metrics.** Every request costs LLM calls;
  put the service behind a gateway that authenticates and rate-limits it.
