# RAGU decision matrix

Every signature below is verified against the repository source. Take class names and
parameters from here.

---

# Part 1. Questions

Wording is ready to use verbatim in `AskUserQuestion` (translate into the user's
language). Library terms never appear in the question text — they live in the "choice"
column.

## Q1. Build a graph, or not *(always, and always first)*

Asked directly, in library terms, because it is the one decision that dominates cost and
because most people setting up RAGU already have a view on it. Give the price in the
question so the answer is an informed one.

> **Строить граф знаний или ограничиться векторным индексом по чанкам?**
> Граф — это прогон LLM по всему корпусу на извлечение сущностей и связей: дороже и
> дольше на сборке, зато отвечает на вопросы, которые надо склеивать из разных мест.
> Векторный индекс — только эмбеддинги чанков: минуты вместо часов, но каждый ответ
> ограничен тем, что нашлось в одном фрагменте.
> - Граф
> - Векторный индекс
> - Не знаю — спроси иначе

| Answer | Choice | Consequences |
|---|---|---|
| graph | extractor + `LocalSearchEngine` | Q6 needed |
| vector index | `BuilderArguments(build_only_vector_context=True)`, no extractor, `NaiveSearchEngine` | Q6 not asked, graph parameters unused |
| not sure | ask **Q1b** and decide from that | — |

## Q1b. How answers are laid out *(only when Q1 = "not sure")*

The same decision reached from behaviour instead of terminology. Never ask this when Q1
already settled it — it is the same question twice.

> **How is the answer to your questions usually laid out?**
> - "It sits in one place in the text — you just find the right passage and quote it"
> - "It has to be assembled from several places: who is connected to whom, what follows
>   what, who worked where"
> - "Both, depending on the question"

| Answer | Choice | Consequences |
|---|---|---|
| one passage | flat index, as Q1 = vector index | Q6 not asked; indexing is far cheaper and faster |
| assembled from several places | graph, as Q1 = graph | Q6 needed |
| both | graph + `MixSearchEngine([Local, Naive])` | more expensive, covers both modes |

## Q2. Query type → engine *(always)*

> **What will you be asking? Pick everything that looks familiar:**
> - "What year was company X founded?", "Who signed the document?" — pinpoint facts
> - "What are these documents about overall, what themes and problems recur?" — corpus overview
> - "Which of the people mentioned worked at companies founded before 1990?" — multi-hop
> - "Find the clause about late-payment penalties" — locate and quote a passage

Multi-select. Mapping:

| Selected | Engine |
|---|---|
| pinpoint facts | `LocalSearchEngine` |
| corpus overview | `GlobalSearchEngine` + `BuilderArguments(make_community_summary=True)` |
| locate a passage | `NaiveSearchEngine` |
| multi-hop | `QueryPlanEngine(engine=<primary engine>)` — a wrapper, not a replacement |
| 2 or more selected | `MixSearchEngine(llm, engines=[...])` |

`GlobalSearchEngine` needs community summaries to exist, so it is impossible without a
graph. If Q1 ruled the graph out but "corpus overview" is selected here, that is a
contradiction — go back to Q1 and say so plainly.

## Q3. Lexical retrieval *(always)*

> **Do your queries contain exact strings that must be matched literally — part numbers,
> standard or statute references, model names, surnames, error codes?**
> - Yes, regularly
> - No, they are ordinary prose questions

"Yes" → sparse embedder `BM25` (or `BM42`), passed both into `KnowledgeGraph` and into
the engine. The reason: dense embeddings handle rare tokens poorly.

## Q4. Models *(always)*

> **Where do the models run?**
> - A hosted OpenAI-compatible API
> - Your own vLLM / Ollama / TGI server behind an OpenAI-compatible endpoint
> - Locally on this machine via transformers
> - Not decided yet

The first two are the same code and differ only in `base_url`. The third only concerns
the embedder, reranker and chunker: **RAGU's LLM always goes through an OpenAI-compatible
client**, so a local model still needs a server (vLLM with the OpenAI API).

Do not go hunting through the project for credentials or endpoints — ask.

## Q5. Size and updates *(always)*

> **How many documents, and what happens to them next?**
> - Up to a few hundred, built once
> - Thousands, and the corpus keeps growing
> - Tens of thousands or more / needs access from several processes

| Answer | Backends |
|---|---|
| hundreds, one-off | defaults: `NetworkXStorage` + `NanoVectorDBStorage` + `JsonKVStorage` (plain files, nothing to deploy) |
| thousands, growing | `NetworkXStorage` + `QdrantVectorDBStorage` (local path or server) |
| tens of thousands / multi-process | `Neo4jStorage` + `QdrantVectorDBStorage` (server) |

Rule of thumb for leaving networkx: it holds the whole graph in memory and rewrites the
entire `.gml` on every `index_done_callback` — fine to roughly 10⁵ nodes, painful beyond.

## Q6. Extraction quality vs cost *(only if Q1 = graph)*

> **The first build runs an LLM across the whole corpus. What matters more?**
> - Best possible quality, cost is not the constraint
> - A reasonable balance
> - As cheap as possible (the corpus is in Russian)

| Answer | Extractor |
|---|---|
| quality | `TwoStageArtifactsExtractorLLM(..., do_entity_validation=True, do_relation_validation=True)` + `ICLConfig(enabled=True)` |
| balance | `ArtifactsExtractorLLM(..., do_validation=True, icl_config=ICLConfig(enabled=True, num_examples=2))` |

## Q7. Source formats *(only if not inferred from the data directory)*

**RAGU ingests plain text only.** There is no parser layer: no PDF, DOCX, HTML, image,
audio or video ingestion, no OCR, no ASR. Documents reach the pipeline as strings.

So there is nothing to choose here, and usually nothing to ask. Phase 0 already listed the
file extensions; the only thing to do with them is check that the corpus is text:

| What's in the data | What to do |
|---|---|
| `.txt`, `.md` | `read_text_from_files(directory, file_extensions={".txt", ".md"})` |
| anything else | say so plainly: RAGU cannot read it. The user has to convert to text first — that conversion is outside this build and outside RAGU |

If a corpus is mixed, index the text part and name the excluded files in the summary and
in `ragu_build.yaml`, so the gap is recorded rather than silently dropped.

## Q8. Corpus language *(if not obvious)*

`Settings.language = "russian" | "english"`. Drives prompts and answer language.


## Optional add-ons

Ask only if the user raises quality or latency themselves:

- **Reranker** — `ScorerOpenAI` / `ScorerCrossEncoder`, via the `reranker=` parameter on
  Local/Naive, plus `rerank_top_k` in the query params.
- **Streaming answers** — `engine.stream_query(...)`, an async iterator.
- **Batched queries** — `batch_query` / `batch_search`, position-aligned with the input,
  fail-fast.
- **LLM response cache** — `Settings.cache_path = "./cache"`, saves money on reruns.

---

# Part 2. Components: exact signatures

## Global settings

```python
from ragu.common.global_parameters import Settings   # an INSTANCE, not a class

Settings.language = "russian"                  # "english" by default
Settings.storage_folder = "./ragu_working_dir/my_build"
Settings.cache_path = "./cache"                # None => caching disabled
Settings.llm_context_token_limit = 30_000
Settings.embedder_token_limit = 8_192
Settings.tokenizer_llm_backend = "tiktoken"    # "tiktoken" | "local"
Settings.tokenizer_llm_name = "gpt-4o"
Settings.tokenizer_embedder_name = "text-embedding-3-large"
```

**Persistence gotcha:** `storage_folder` defaults to `ragu_working_dir/<timestamp>`, so
every run would write to a fresh directory. To reuse a built graph, set `storage_folder`
explicitly and identically in both the build script and the query script — the
file-backed stores pick up existing files on initialization.
For local tokenizers (`"local"`), set `tokenizer_llm_name` to the HF model name.

## Client, LLM, embedder

```python
from ragu.models.openai import CachedAsyncOpenAI
from ragu.models.llm import LLMOpenAI
from ragu.models.embedder import EmbedderOpenAI

client = CachedAsyncOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),   # None => api.openai.com; vLLM => http://host:8000/v1
    api_key=os.getenv("OPENAI_API_KEY"),     # vLLM usually accepts any non-empty value
    rate_max_simultaneous=10,                # lower this for a small server
    rate_max_per_minute=100,
    # rate_min_delay=..., retry_times_sec=(2, 4, 8)
)

llm = LLMOpenAI(client=client, model_name=os.getenv("LLM_MODEL_NAME"))

embedder = EmbedderOpenAI(
    client=client,
    model_name=os.getenv("EMBEDDER_MODEL_NAME"),
    dim=None,              # None => probed during initialize()
    batch_size=500,
    max_concurrent_batches=5,
)
await embedder.initialize()   # REQUIRED before use
```

`CachedAsyncOpenAI` also accepts a ready `client=AsyncOpenAI(...)`, in which case
`base_url`/`api_key` are ignored. Only transient errors are retried.

## Sparse embedder (lexical)

```python
from ragu.models.sparse_embedder import BM25   # or BM42

sparse_embedder = BM25(model_name="Qdrant/bm25", language="russian")
```

It must go in **both** places or there is no hybrid:

```python
kg = KnowledgeGraph(..., sparse_embedder=sparse_embedder)
engine = LocalSearchEngine(..., sparse_embedder=sparse_embedder)
```

With `QdrantVectorDBStorage`, also set `sparse_type="bm25"` (or `"bm42"`) in
`vdb_storage_kwargs`.

## Reranker

```python
from ragu.models.scorer import ScorerOpenAI, ScorerCrossEncoder

reranker = ScorerOpenAI(client=client, model_name="<rerank-model>")
# or locally:
# from sentence_transformers import CrossEncoder
# reranker = ScorerCrossEncoder(model=CrossEncoder("BAAI/bge-reranker-v2-m3"), batch_size=16)
```

## Chunkers

```python
from ragu.chunker import SimpleChunker, SemanticTextChunker, SmartSemanticChunker

SimpleChunker(max_chunk_size=1000, overlap=0)                    # characters; the default choice
SemanticTextChunker(model_name="...", max_chunk_size=512, device="cuda:0")
SmartSemanticChunker(reranker_name="BAAI/bge-reranker-v2-m3", max_chunk_length=250, device="cuda:0")
```

These three are all there is. Custom splitting means subclassing `BaseChunker`.

The semantic chunkers pull in torch/sentence-transformers and want a GPU — do not propose
them if phase 0 found none.

## Extractors

```python
from ragu import ArtifactsExtractorLLM, TwoStageArtifactsExtractorLLM, RaguLmArtifactExtractor
from ragu.common.prompts import ICLConfig

extractor = ArtifactsExtractorLLM(
    llm=llm,
    embedder=embedder,                 # required for ICL
    icl_config=ICLConfig(enabled=True, num_examples=2, selection_strategy="hybrid"),
    do_validation=True,
    language=None,                     # None => Settings.language
    # entity_types=..., relation_types=...   # NEREL types by default
)

extractor = TwoStageArtifactsExtractorLLM(
    llm=llm, embedder=embedder, icl_config=...,
    do_entity_validation=True, do_relation_validation=True,
)

extractor = RaguLmArtifactExtractor(llm=llm, temperature=0.0, top_p=0.95)
```

Custom entity/relation types go through `entity_types=` / `relation_types=`; the defaults
are the NEREL lists in `ragu/triplet/types.py`.

## Graph build parameters

```python
from ragu import BuilderArguments

BuilderArguments(
    use_llm_summarization=True,      # merge duplicate descriptions via LLM
    use_clustering=False,            # cluster descriptions inside ArtifactsSummarizer
                                     # before merging them. NOT community detection —
                                     # that is driven by make_community_summary alone.
                                     # Needs an embedder, and only kicks in above
                                     # cluster_only_if_more_than entities.
    build_only_vector_context=True,  # <-- FLAT INDEX: extraction is skipped entirely
    make_community_summary=True,     # required by GlobalSearchEngine
    remove_isolated_nodes=True,
    cluster_only_if_more_than=10_000,
    summarize_only_if_more_than=7,
    max_cluster_size=128,
    min_cluster_size=1,              # >1 drops small communities before summarization
    random_seed=42,
)
```

`vectorize_chunks` is a no-op kept for API stability; chunks are always vectorized.

## Storage backends

```python
from ragu import StorageArguments
from ragu.storage.graph_storage_adapters import NetworkXStorage
from ragu.storage.kv_storage_adapters.json_storage import JsonKVStorage
from ragu.storage.vdb_storage_adapters.nano_vdb import NanoVectorDBStorage

# Defaults need no arguments at all:
storage_settings = StorageArguments()
```

Production variant:

```python
from ragu.storage.graph_storage_adapters.neo4j_adapter import Neo4jStorage  # explicit import only
from ragu.storage.vdb_storage_adapters.qdrant_vdb import QdrantVectorDBStorage

storage_settings = StorageArguments(
    graph_backend_storage=Neo4jStorage,
    vdb_storage_type=QdrantVectorDBStorage,
    graph_storage_kwargs={"uri": "bolt://localhost:7687", "user": "neo4j",
                          "password": "...", "database": "neo4j"},
    vdb_storage_kwargs={"url": "http://localhost:6333", "sparse_type": "bm25"},
)
```

How this wires up: `Index` supplies `filename` (an absolute path inside
`Settings.storage_folder`), `embedding_dim` (from `embedder.dim`) and `node_cls`/`edge_cls`
itself. Your kwargs go on top. Never set `embedding_dim` by hand — if it disagrees with
the embedder, `Index` raises `ValueError`.

- `NetworkXStorage(filename, node_cls, edge_cls)` — `.gml`, entirely in memory.
- `Neo4jStorage(uri, user, password, node_cls, edge_cls, database="neo4j")` — needs the
  `neo4j` driver, which is why it is not re-exported from the package.
- `NanoVectorDBStorage(embedding_dim, score_threshold=None, storage_folder=..., filename=..., metric="cosine")`
- `QdrantVectorDBStorage(embedding_dim, storage_folder=None, filename=..., collection_name=None,
  path=None, url=None, host=None, port=None, api_key=None, sparse_type=None, ...)` —
  without `url`/`host` it runs in local on-disk mode at `path`.

Long-lived processes should `await index.close()` when done, or Neo4j/Qdrant connections
leak.

## Graph

```python
from ragu import KnowledgeGraph

kg = KnowledgeGraph(
    llm=llm,
    embedder=embedder,
    sparse_embedder=None,
    chunker=chunker,
    artifact_extractor=extractor,      # None when build_only_vector_context=True
    builder_settings=builder_settings,
    storage_settings=storage_settings,
    additional_modules=None,           # list of GraphBuilderModule
    language=None,
)
await kg.build_from_docs(docs)         # List[str] — plain text, nothing else
```

CRUD and maintenance (all async): `upsert_entities`, `update_entities`, `get_entities`,
`delete_entities`, the same for relations/communities/summaries, plus `get_chunks`,
`reindex_community`, `reindex_descriptions`, `reindex_graph`.
Calling `build_from_docs` again deduplicates chunks by id — that is how incremental
ingestion works.

## Engines

```python
from ragu import (LocalSearchEngine, GlobalSearchEngine, NaiveSearchEngine,
                  MixSearchEngine, QueryPlanEngine)

LocalSearchEngine(llm, kg, embedder, sparse_embedder=None, reranker=None,
                  language=None, max_context_length=None,
                  tokenizer_backend=None, tokenizer_model=None)

NaiveSearchEngine(llm, kg, embedder, sparse_embedder=None, reranker=None, ...)   # same signature

GlobalSearchEngine(llm, kg, language=None, max_context_length=None, ...)         # no embedder

MixSearchEngine(llm, engines=[local, naive], engine_params=None,
                allow_partial_failures=True, ...)

QueryPlanEngine(engine=local)          # wraps any engine
```

Calls are async, except streaming which returns an async iterator:

```python
response = await engine.query("question")        # response.response is the text
retrieval = await engine.search("question")      # context only, no generation
answers = await engine.batch_query(["q1", "q2"]) # position-aligned, fail-fast
async for event in engine.stream_query("question"):
    ...
```

`a_query` / `a_search` are legacy aliases — do not use them in new code.

Query parameters are passed as objects:

```python
from ragu.search_engine.local_search import LocalParams
from ragu.search_engine.naive_search import NaiveSearchParams
from ragu.search_engine.global_search import GlobalSearchParams
from ragu.search_engine.mix_search import MixQueryParams

LocalParams(top_k=20, rerank_top_k=None, use_summary=False, use_chunks=True)
NaiveSearchParams(top_k=20, rerank_top_k=None)
GlobalSearchParams(min_cluster_size=1)
MixQueryParams(ensemble_responses=False)   # True => ensemble answers rather than contexts
```

## Loading documents

One way in: a directory of text files.

```python
from ragu.utils.ragu_utils import read_text_from_files

docs = read_text_from_files("path/to/dir", file_extensions={".txt", ".md"})
```

It walks the directory recursively and skips anything it cannot decode as UTF-8. Passing
`file_extensions=None` reads every file, which on a mixed directory means silently
skipping the binaries — always pass the set explicitly so the selection is visible.

Anything that is not already text has to be converted before it gets here. That
conversion is the user's job and lives outside RAGU.

---

# Part 3. Stock builds

Ready-made combinations to start from. `assets/build_template.py` is build B written out
in full — start there and swap parts.

**A. Quick start, flat index.**
`SimpleChunker(1000)` + `BuilderArguments(build_only_vector_context=True)` +
`StorageArguments()` + `NaiveSearchEngine`. No extractor; indexing costs embeddings only.

**B. Classic graph (the default).**
`SimpleChunker(1000)` + `ArtifactsExtractorLLM(do_validation=True, icl_config=...)` +
`BuilderArguments()` + `StorageArguments()` + `LocalSearchEngine`.
This is `examples/extract_with_llm_and_local_search.py`.

**C. Graph plus overview questions.**
B + `use_clustering=True, make_community_summary=True` +
`MixSearchEngine(llm, [LocalSearchEngine(...), GlobalSearchEngine(...)])`.

**D. Hybrid with lexical retrieval.**
B + `BM25` in both `KnowledgeGraph` and the engine + a reranker.

**E. Production.**
D + `Neo4jStorage` + `QdrantVectorDBStorage(sparse_type="bm25")` + `Settings.cache_path`
+ an explicit `Settings.storage_folder`.

**F. Multi-hop questions.**
Any of B–E wrapped in `QueryPlanEngine(engine=...)`.
