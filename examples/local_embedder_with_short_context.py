"""
Example: building and querying a knowledge graph with a local embedding model
that has a short context window (e.g. 512 tokens).

This demonstrates the ``Settings`` configuration for tokenizer and token limits,
as well as separate ``CachedAsyncOpenAI`` clients for the LLM and the local
embedder served via `vLLM <https://github.com/vllm-project/vllm>`_.

Choosing an embedder for your language
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- **English only**: ``BAAI/bge-large-en-v1.5`` (512 tokens).
- **Multilingual / Russian**: ``intfloat/multilingual-e5-large`` (512 tokens).

Adjust ``Settings.embedder_token_limit`` to match the model's context window.
A smaller or less capable embedder may require a larger ``top_k`` in
``a_query()`` to compensate for lower retrieval precision.

Usage
-----
1. Start the local embedder with vLLM.
   Use ``--enforce-eager`` to avoid CUDA assertion errors with some
   models (e.g. ``intfloat/multilingual-e5-large``)::

        vllm serve intfloat/multilingual-e5-large --port 8001 --enforce-eager

2. Export environment variables::

       export OPENAI_BASE_URL="https://..."       # LLM API endpoint
       export OPENAI_API_KEY="sk-..."
       export LLM_MODEL_NAME="gpt-4o-mini"
       export LOCAL_EMBEDDER_URL="http://localhost:8001/v1"
       export EMBEDDER_MODEL_NAME="intfloat/multilingual-e5-large"

3. Run::

       python examples/local_embedder_with_short_context.py
"""

import asyncio
import os

from ragu import (
    ArtifactsExtractorLLM,
    BuilderArguments,
    KnowledgeGraph,
    LocalSearchEngine,
    Settings,
    SimpleChunker,
)
from ragu.common.prompts import ICLConfig
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI
from ragu.utils.ragu_utils import read_text_from_files


async def main():
    # ── Global settings ─────────────────────────────────────────────
    Settings.storage_folder = "ragu_working_dir/local_embedder_example"
    Settings.language = "russian"

    # Configure token limits and tokenizer for the local embedder.
    # These defaults are used by EmbedderOpenAI unless overridden per-instance.
    #
    # IMPORTANT: choose an embedder that matches your language:
    #   English  → BAAI/bge-large-en-v1.5   (512 tokens)
    #   Multilingual / Russian → intfloat/multilingual-e5-large (512 tokens)
    #
    # Set embedder_token_limit to match the model's context window.
    Settings.embedder_token_limit = 512
    Settings.tokenizer_embedder_backend = "local"
    Settings.tokenizer_embedder_name = os.getenv("EMBEDDER_MODEL_NAME")

    # ── Load documents ──────────────────────────────────────────────
    docs = read_text_from_files("examples/data/ru")

    chunker = SimpleChunker(max_chunk_size=1000)

    # ── LLM client (remote API) ────────────────────────────────────
    llm_client = CachedAsyncOpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        rate_max_simultaneous=10,
        rate_max_per_minute=100,
    )

    llm = LLMOpenAI(
        client=llm_client,
        model_name=os.getenv("LLM_MODEL_NAME"),
    )

    # ── Embedder client (local vLLM) ───────────────────────────────
    embed_client = CachedAsyncOpenAI(
        base_url=os.getenv("LOCAL_EMBEDDER_URL"),
        api_key="unused",
        rate_max_simultaneous=20,
        rate_max_per_minute=500,
        embed_timeout=60.0,
    )

    # batch_size=32 and max_concurrent_batches=2 are tuned for a local GPU.
    # Defaults (500 and 5) are designed for cloud APIs and may cause OOM on vLLM.
    embedder = EmbedderOpenAI(
        client=embed_client,
        model_name=os.getenv("EMBEDDER_MODEL_NAME"),
        embedder_token_limit=512,
        tokenizer_backend="local",
        tokenizer_name=os.getenv("EMBEDDER_MODEL_NAME"),
        batch_size=32,
        max_concurrent_batches=2,
    )
    await embedder.initialize()

    # ── Build knowledge graph ───────────────────────────────────────
    icl_config = ICLConfig(
        enabled=True,
        num_examples=2,
        selection_strategy="hybrid",
    )

    artifact_extractor = ArtifactsExtractorLLM(
        llm=llm,
        embedder=embedder,
        icl_config=icl_config,
        do_validation=True,
    )

    builder_settings = BuilderArguments(
        use_llm_summarization=True,
    )

    knowledge_graph = KnowledgeGraph(
        llm=llm,
        embedder=embedder,
        chunker=chunker,
        artifact_extractor=artifact_extractor,
        builder_settings=builder_settings,
    )
    await knowledge_graph.build_from_docs(docs)

    # ── Query ───────────────────────────────────────────────────────
    search_engine = LocalSearchEngine(
        llm=llm,
        knowledge_graph=knowledge_graph,
        embedder=embedder,
    )

    questions = [
        "Кто написал гимн Норвегии?",
        "Шум, издаваемый ЭТИМИ ПАУКООБРАЗНЫМИ, слышен за пять километров. Отсюда и их название.",
        "Как переводится название романа 'Ка́мо гряде́ши, Го́споди?' на русский языке"
    ]

    for question in questions:
        print(f"\nQ: {question}")
        # top_k=40 compensates for lower retrieval precision of smaller embedders
        answer = await search_engine.a_query(question, top_k=40)
        print(f"A: {answer.response}")


if __name__ == "__main__":
    asyncio.run(main())
