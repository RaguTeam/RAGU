# Examples

This directory contains example scripts demonstrating RAGU usage.

## extract_with_llm_and_local_search.py

End-to-end example that builds a knowledge graph from Russian-language text files and performs local search queries.

### What it does

1. Loads `.txt` files from `data/ru/`
2. Chunks documents with `SimpleChunker`
3. Extracts entities and relations using `ArtifactsExtractorLLM` with in-context learning (few-shot examples selected by semantic similarity)
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
