# RAGU search service

HTTP service in front of a prebuilt knowledge graph. It owns the graph; clients
only search it. One route per search mode, so a gateway can apply different
timeouts and rate limits per mode without touching the service — and they do
differ: `global` rates every community summary against the query with its own
LLM call (N+1 calls for N communities, growing with the graph), while `local`
and `naive` issue one generation each. Give `/v1/search/global` the long timeout
and the tight rate limit; `global_min_cluster_size` is the knob that cuts N.

The service ships with the package as `ragu.api`, so an installed RAGU can
serve a graph without a checkout.

The service **serves** a graph, it never builds one. Build the graph first (for
example with `build_vector_index.py --build`), then point the service at that
storage folder.

## Run

```bash
# Canned answers, no graph and no LLM required — for developing clients
RAGU_API_BACKEND=stub python -m ragu.api --port 8020

# Real graph
RAGU_API_STORAGE_FOLDER=./ragu_working_dir \
RAGU_API_LANGUAGE=russian \
python -m ragu.api --port 8020
```

LLM and embedder credentials come from the same `.env` the rest of RAGU uses
(`ragu.common.env.Env`): `LLM_MODEL_NAME`, `LLM_BASE_URL`, `LLM_API_KEY`, and
optionally `EMBEDDER_BASE_URL`, `EMBEDDER_API_KEY`, `EMBEDDER_MODEL_NAME`.
Service-level settings are `RAGU_API_*` environment variables — see
`ServiceSettings` in `ragu/api/config.py`; `--host/--port/--backend/--storage-folder`
override them on the command line.

`RAGU_API_STORAGE_FOLDER` must be set explicitly and point at one graph
directory, not at a folder of them: the default `Settings.storage_folder` is
timestamped per run and would point at an empty directory.

## Docker

```bash
docker build -t ragu-api .                          # from the repository root
docker compose up -d ragu-api
docker compose --profile qdrant up -d               # add a remote Qdrant
```

The build context is whitelisted in `.dockerignore` — the repository root holds
multi-gigabyte working directories that must stay out of it.

- Dependencies come from `uv.lock` in one step (`uv sync --frozen --no-dev
  --no-editable --extra api`), not from a fresh resolve: `openai>=2.32.0`
  resolves to 3.x, where `openai._utils._logs.httpx_logger` no longer exists and
  `ragu.common.logger` fails on import. The lock pins the version that works.
- `--build-arg RAGU_SYNC_EXTRAS="--extra local"` adds sentence-transformers and
  transformers for local rerankers and embedders; the default image stays around
  1 GB of site-packages.
- The tiktoken vocabularies are baked at build time; otherwise the first request
  downloads them and an offline deployment fails outright.
- The healthcheck requires `"graph_loaded":true`, so `depends_on:
  service_healthy` waits for a service that can actually answer searches. Large
  graphs take minutes to load (a 200 MB GML plus 170k relation vectors is a few
  minutes on a warm page cache), hence the 10-minute start period.
- Set `RAGU_API_EMBEDDER_DIM` to the dimension the graph was built with. Without
  it the service sends one probe embedding at startup to detect the dimension,
  and startup fails if the embedder endpoint is unreachable. The dimension is
  stored in the graph: `head -c 40 <graph>/vdb_entity.json` shows
  `{"embedding_dim": N, ...`.
- The graph directory is mounted read-write at `/data/graph` and must be
  writable by uid 1000 (the `ragu` user in the image): the storages open missing
  files for writing at construction, so a read-only or root-owned mount fails
  the load with `PermissionError` rather than serving the graph.
- With the Docker **snap**, the daemon has a private `/tmp` and no access to it
  from the host, so a graph path under `/tmp` silently mounts as an empty
  directory. Keep graphs under `$HOME` (or use a named volume).
- The LLM cache lives in a named volume.

## API

| Route                    | Body                                                        |
|--------------------------|-------------------------------------------------------------|
| `POST /v1/search/global` | `query`, `params: GlobalSearchParams`                         |
| `POST /v1/search/local`  | `query`, `use_query_plan=true`, `params: LocalParams`        |
| `POST /v1/search/naive`  | `query`, `use_query_plan=true`, `params: NaiveSearchParams`  |
| `GET /health`            | `{"status": "ok", "graph_loaded": true}`                     |

`params` is the engine's own parameter class, embedded in the request model
rather than restated field by field:

```json
{"query": "кто написал роман?", "use_query_plan": true,
 "params": {"top_k": 20, "rerank_top_k": null, "use_summary": true, "use_chunks": true}}
```

Two consequences of that choice:

- **Defaults are the engine's.** Omitting `params` yields `LocalParams()` /
  `NaiveSearchParams()`, so `use_summary` is False and `top_k` is 20 — not the
  5 the service used to default to. Clients that care should send the values.
- **Bounds are the service's.** The dataclasses carry none, so `top_k` is
  clamped to `RAGU_API_MAX_TOP_K` (default 100).
- **Misplaced fields are rejected.** The request models forbid extras, so a
  top-level `top_k` (the shape before `params` existed) answers 400 instead of
  quietly serving the engine default. Unknown keys *inside* `params` are still
  ignored — that is pydantic's behaviour for stdlib dataclasses.

All three modes carry their engine's parameter class: `GlobalSearchParams`
(`min_cluster_size`, which skips communities smaller than the bound),
`LocalParams` and `NaiveSearchParams`.

Global search takes no `use_query_plan`: its answer is synthesized from
summaries that already cover the whole corpus, so decomposing the question would
only re-rate the same summaries once per subquery. `used_query_plan` stays in
the response envelope and is always `false` for that mode.

Success response:

```json
{
  "query": "Кто написал роман 'Камо грядеши'?",
  "mode": "local",
  "used_query_plan": true,
  "answer": "Роман написал Генрик Сенкевич...",
  "sources": [{"id": "chunk-8f3...", "type": "chunk", "content": "...", "score": 0.87}],
  "subqueries": [{"query": "Кто написал роман?", "answer": "Генрик Сенкевич"}]
}
```

Errors share one envelope:

```json
{"error": {"code": "CAPABILITY_UNAVAILABLE", "mode": "global",
           "missing_capability": "community_summaries", "message": "..."}}
```

| Status | Code                     | Situation                                        |
|--------|--------------------------|--------------------------------------------------|
| 400    | `INVALID_REQUEST`        | empty `query`, bad field type                     |
| 409    | `CAPABILITY_UNAVAILABLE` | the graph cannot serve this mode                  |
| 503    | `SERVICE_NOT_READY`      | graph not loaded, empty chunk storage             |
| 500    | `INTERNAL_ERROR`         | LLM unavailable, embedder timeout, engine failure |

409 exists so that a client can react: a graph built without community
summaries cannot answer global searches, and a graph built without entity
extraction cannot answer local ones. The status says which capability is
missing and which modes remain.

## Design

`SearchBackend` (`ragu/api/backends/base.py`) is the whole contract: `graph_loaded`,
`startup`, and one method per mode returning a `SearchOutcome`.

- `StubBackend` — deterministic canned answers.
  `RAGU_API_STUB_MISSING_CAPABILITIES=community_summaries,entity_graph,vector_index`
  makes the corresponding modes fail, which is how the 409/503 paths are tested.
- `RaguBackend` — loads a `KnowledgeGraph` from the storage folder and keeps one
  engine per mode.

The engines answer from whatever context they gathered, empty included: a graph
built without community summaries still returns a confident global answer built
on nothing. So the service judges the retrieval instead of the graph — a search
that comes back with no sources at all is reported as `409`, never as an answer.
That covers both an unsuitable graph and a suitable one that had nothing on this
query, and it needs no assumptions about storage internals. The price is one
generation call spent before the emptiness is known.

*TODO:* once the builder records graph metadata (entity/chunk counts,
`make_community_summary`), read it at startup and refuse an unsupported mode up
front instead of paying for that call.

Per-request options travel as `LocalParams` / `NaiveSearchParams` rather than
constructor arguments, because `QueryPlanEngine` forwards `params` to the
wrapped engine. Those same classes are the request schema, so an engine
parameter added upstream is available over HTTP without a change here — and one
renamed or removed upstream changes the wire contract, which is the price of
not restating them.

Engine results are flattened into `{id, type, content, score}`: chunks and their
scores from `NaiveSearchResult`, entities/relations/summaries/chunks from
`LocalSearchResult`, rated insights from `GlobalSearchResult`. A result type the
adapter does not recognise degrades to a single source rendered with
`to_text()`. Query-plan subqueries are read from
`SearchEngineResponse.payload`, which maps subquery ids to their responses.

## Tests

```bash
pytest tests/api
```
