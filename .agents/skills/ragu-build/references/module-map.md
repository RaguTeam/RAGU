# RAGU repository map

Where to look when the decision matrix is not enough. Read **narrowly**: one README per
open question, never everything at once.

## Modules and their READMEs

RAGU takes plain text in. There is no parser, OCR or ASR layer — do not look for one and
do not propose one.

| Module | README | Answers |
|---|---|---|
| `ragu/chunker/` | `ragu/chunker/README.md` | how text is split |
| `ragu/models/` | `ragu/models/README.md` | LLM, embedder, sparse embedder, reranker, response cache, rate limits |
| `ragu/triplet/` | `ragu/triplet/README.md` | entity and relation extraction, NEREL types, ICL |
| `ragu/graph/` | `ragu/graph/README.md` | build pipeline, summarization, communities, `Index`, incremental updates |
| `ragu/storage/` | `ragu/storage/README.md` | backends: networkx/neo4j, nano_vdb/qdrant, json kv |
| `ragu/search_engine/` | `ragu/search_engine/README.md` | search engines, batch mode, streaming, query planner |
| `ragu/common/` | `ragu/common/README.md` | `Settings`, prompts, cache, logger |
| `ragu/utils/` | `ragu/utils/README.md` | reading text files, token truncation, text normalization |

The full package tree is in `ragu/README.md`.

## Examples (known-good working builds)

Cite these in your report — a ready-made notebook is more useful to the user than a
paraphrase.

| File in `examples/` | Shows |
|---|---|
| `extract_with_llm_and_local_search.py` | **the canonical build**: graph + LocalSearch; model generated scripts on this |
| `search_engines_example.ipynb` | Local / Global / Naive / Mix compared on the same data |
| `hybrid_local_search_example.ipynb` | wiring a sparse embedder, hybrid retrieval |
| `mix_search_example.ipynb` | ensembling engines |
| `query_plan_example.ipynb` | decomposing complex multi-hop queries |
| `neo4j_qdrant_setup_example.ipynb` | production storage backends |
| `graph_adapters_example.ipynb`, `vector_adapters_example.ipynb` | swapping one backend at a time |
| `incremental_updates_example.ipynb` | adding documents to an existing graph |
| `batch_operations_example.ipynb`, `batch_retrieval_benchmark.ipynb` | batched queries, benchmarking |
| `llm_embedder_reranker_example.ipynb` | reranker together with an engine |
| `local_embedder_with_short_context.py` | local embedder with a short context window |
| `knowledge_graph_settings_example.ipynb` | `Settings`, working directory, persistence |
| `custom_prompts_example.ipynb` | replacing prompts |
| `streaming_example.ipynb` | `stream_query` |
| `llm_as_judge_example.ipynb` | evaluating answer quality |

## Finding the actual signatures

If a parameter appears in neither the matrix nor a README:

```bash
grep -n "def __init__" -A 25 ragu/<module>/<file>.py
grep -n "class <ClassName>" -A 40 ragu/<module>/<file>.py
```

Top-level public exports live in `ragu/__init__.py`. Anything listed there imports as
`from ragu import X`; everything else needs its full module path.
