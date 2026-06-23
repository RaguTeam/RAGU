# Examples

This directory contains example scripts demonstrating RAGU usage.

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
