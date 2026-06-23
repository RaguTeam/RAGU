# RAGU Package Tree

```text
ragu/
├── __init__.py
├── README.md
├── chunker/
│   ├── __init__.py
│   ├── base_chunker.py
│   ├── chunkers.py
│   ├── types.py
│   └── README.md
├── common/
│   ├── __init__.py
│   ├── base.py
│   ├── batch_generator.py
│   ├── cache.py
│   ├── env.py
│   ├── global_parameters.py
│   ├── logger.py
│   └── prompts/
│       ├── __init__.py
│       ├── default_models.py
│       ├── default_templates.py
│       ├── few_shot.py
│       ├── icl_config.py
│       ├── icl_examples/
│       ├── icl_manager.py
│       ├── messages.py
│       ├── prompt_storage.py
│       └── README.md
├── graph/
│   ├── __init__.py
│   ├── artifacts_summarizer.py
│   ├── builder_modules.py
│   ├── community_summarizer.py
│   ├── graph_builder_pipeline.py
│   ├── graph_retrieve_backend.py
│   ├── index.py
│   ├── knowledge_graph.py
│   ├── types.py
│   └── README.md
├── models/
│   ├── __init__.py
│   ├── caching.py
│   ├── embedder.py
│   ├── llm.py
│   ├── openai.py
│   ├── README.md
│   ├── scorer.py
│   └── sparse_embedder.py
├── search_engine/
│   ├── __init__.py
│   ├── base_engine.py
│   ├── global_search.py
│   ├── local_search.py
│   ├── mix_search.py
│   ├── naive_search.py
│   ├── query_plan.py
│   ├── search_functional.py
│   └── README.md
├── storage/
│   ├── __init__.py
│   ├── base_storage.py
│   ├── types.py
│   ├── README.md
│   ├── graph_storage_adapters/
│   │   ├── README.md
│   │   └── networkx_adapter.py
│   ├── kv_storage_adapters/
│   │   ├── README.md
│   │   └── json_storage.py
│   └── vdb_storage_adapters/
│       ├── README.md
│       ├── nano_vdb.py
│       └── qdrant_vdb.py
├── triplet/
│   ├── __init__.py
│   ├── base_artifact_extractor.py
│   ├── llm_artifact_extractor.py
│   ├── prompts.py
│   ├── ragu_lm_artifact_extractor.py
│   ├── two_stage_extractor.py
│   ├── types.py
│   └── README.md
└── utils/
    ├── __init__.py
    ├── ragu_utils.py
    ├── text_normalize.py
    ├── token_truncation.py
    ├── README.md
    └── testing/
        ├── __init__.py
        ├── openai_mock_server.py
        └── README.md
```

## What the folders contain

`chunker/`
Chunking logic and chunk data types.

`common/`
Shared settings, prompt helpers, cache utilities, logging, and base classes.

`graph/`
Graph construction, summarization, indexing, and the `KnowledgeGraph` facade.

`models/`
LLM, embedder, scoring, caching, and OpenAI client adapters.

`search_engine/`
Retrieval engines.

`storage/`
Graph, key-value, and vector storage contracts plus concrete adapters.

`triplet/`
Entity and relation extraction modules.

`utils/`
General utilities and testing helpers.
