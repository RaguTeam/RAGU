# Examples

This directory contains example scripts demonstrating RAGU usage.

## Streaming search examples

Real streaming examples for every search engine. Each script can run as an
interactive example or as a one-shot smoke test with `--query`.

### What it does

1. Loads OpenAI-compatible credentials and model names from environment variables
2. Loads the persisted test knowledge graph from `tests/kg_for_test`
3. Calls the engine's `stream_query`
4. Prints streamed answer deltas and fails if no streamed text is produced

### Prerequisites

- RAGU installed (`pip install -e .`)
- An OpenAI-compatible API endpoint with the following environment variables set:
  - `OPENAI_BASE_URL` — API base URL; optional, defaults to `https://api.openai.com/v1`
  - `OPENAI_API_KEY` — API key
  - `LLM_MODEL_NAME` — chat model name
  - `EMBEDDER_MODEL_NAME` — embedding model name compatible with the persisted graph vectors

```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY="sk-..."
export LLM_MODEL_NAME="gpt-4o-mini"
export EMBEDDER_MODEL_NAME="text-embedding-3-large"
```

The default graph path is `tests/kg_for_test`. Its saved vector stores are
3072-dimensional, so the default `--embed-dim` is `3072`. Use an embedding
model with the same output dimension, or pass `--kg-path` and `--embed-dim` for
another persisted graph.

### Running

```bash
python examples/streaming_naive_search_example.py --query "How does naive streaming work?"
python examples/streaming_local_search_example.py --query "How does local search expand context?"
python examples/streaming_global_search_example.py --query "Compare the RAGU search engines."
python examples/streaming_mix_search_example.py --query "How do naive and local search differ?"
python examples/streaming_query_plan_example.py --query "Explain naive search, then explain final streaming."
```

Omit `--query` to enter an interactive prompt. The legacy
`streaming_search_example.py` remains a minimal NaiveSearchEngine-only
interactive example.

If you use another persisted graph, pass its embedding dimension or auto-detect
the live model dimension with one probe request:

```bash
python examples/streaming_naive_search_example.py --kg-path /path/to/kg --embed-dim 0
```

### Scripts

- `streaming_naive_search_example.py` — streams final generation after vector chunk retrieval.
- `streaming_local_search_example.py` — streams final generation after entity, relation, chunk, and optional summary retrieval.
- `streaming_global_search_example.py` — performs structured community meta-evaluation first, then streams final synthesis.
- `streaming_mix_search_example.py` — combines Naive and Local child contexts, then streams final synthesis.
- `streaming_query_plan_example.py` — executes dependency subqueries normally and streams only the final sink subquery answer.

## extract_with_llm_and_local_search.py

End-to-end example that builds a knowledge graph from Russian-language text files and performs local search queries.

### What it does

1. Loads `.txt` files from `data/ru/`
2. Chunks documents with `SimpleChunker`
3. Extracts entities and relations using `ArtifactsExtractorLLM` with in-context learning (few-shot examples selected via a hybrid of semantic similarity and BM25)
4. Builds a knowledge graph with Leiden community detection
5. Runs local search queries against the graph

### Prerequisites

- RAGU installed (`pip install -e .`)
- An OpenAI-compatible API endpoint with the following environment variables set:
  - `OPENAI_BASE_URL` — API base URL
  - `OPENAI_API_KEY` — API key
  - `LLM_MODEL_NAME` — LLM model name (e.g., `mistralai/mistral-medium-3`)
  - `EMBEDDER_MODEL_NAME` — embedding model name (e.g., `emb-qwen/qwen3-embedding-8b`)

```bash
export OPENAI_BASE_URL="https://..."
export OPENAI_API_KEY="sk-..."
export LLM_MODEL_NAME="mistralai/mistral-medium-3"
export EMBEDDER_MODEL_NAME="emb-qwen/qwen3-embedding-8b"
```

### Running

```bash
python examples/extract_with_llm_and_local_search.py
```

### Key configuration points

- **Rate limiting**: The example creates a shared `CachedAsyncOpenAI` with `rate_max_simultaneous=10` and `rate_max_per_minute=100`. For large corpora (thousands of entities/relations), consider using separate clients for LLM and embedder — see the main README ("Client and Rate Limiting Configuration" section).
- **ICL (in-context learning)**: The example enables few-shot example selection via `ICLConfig`. This improves extraction quality by providing the LLM with relevant examples before each extraction call. Four strategies are available: `"semantic"` (default, requires embedder), `"bm25"` (lexical matching, no embedder needed), `"hybrid"` (combines both), and `"random"` (baseline). You can disable it by setting `icl_config=None` or `ICLConfig(enabled=False)`.
- **Language**: Set via `Settings.language`. Examples are filtered to match this language. Supported: `"english"`, `"russian"`.
- **Validation**: Set `do_validation=True` on the extractor to enable a second LLM pass that validates extracted artifacts.

## local_embedder_with_short_context.py

End-to-end example that uses a **local embedding model with a short context window** (e.g., BAAI/bge-large-en-v1.5 with 512 tokens) served via vLLM, alongside a remote LLM API.

### What it does

1. Configures `Settings` with the embedder's token limit (512) and a HuggingFace tokenizer
2. Creates separate `CachedAsyncOpenAI` clients for the LLM (remote) and the embedder (local vLLM)
3. Builds a knowledge graph with automatic text truncation before embedding
4. Runs local search queries

### Prerequisites

- RAGU installed with local tokenizer support (`pip install -e ".[local]"`)
- A local vLLM server serving an embedding model:
  ```bash
  vllm serve intfloat/multilingual-e5-large --port 8001   # Multilingual, 512 tokens
  # or
  vllm serve BAAI/bge-large-en-v1.5 --port 8001           # English only, 512 tokens
  ```
- An OpenAI-compatible LLM API endpoint
- Environment variables:
  - `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `LLM_MODEL_NAME` — LLM configuration
  - `LOCAL_EMBEDDER_URL` — local embedder endpoint (e.g., `http://localhost:8001/v1`)
  - `EMBEDDER_MODEL_NAME` — embedding model name (e.g., `intfloat/multilingual-e5-large` or `BAAI/bge-large-en-v1.5`)

### Key configuration points

- **Choosing an embedder**: Use a multilingual model (e.g. `intfloat/multilingual-e5-large`) for non-English corpora. English-only models (e.g. `BAAI/bge-large-en-v1.5`) produce lower-quality vectors on Russian text, which degrades retrieval precision and can lead to incorrect answers.
- **Token limit**: Set via `Settings.embedder_token_limit` to match the model's context window (512 for both models above). `Settings.tokenizer_embedder_backend = "local"` enables the HuggingFace tokenizer. The LLM tokenizer backend (`Settings.tokenizer_llm_backend`) remains `"tiktoken"` by default and is independent.
- **Retrieval precision**: A smaller or less capable embedder may require a larger `top_k` in `a_query()` (e.g. `top_k=40` instead of the default 20) to compensate for lower vector similarity quality.
