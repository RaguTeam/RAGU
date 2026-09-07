"""RAGU build: knowledge graph + local search.

    python build_demo.py --build     # parse, chunk, extract, index
    python build_demo.py             # ask questions against the existing index

The index lives in STORAGE_FOLDER and is reused across runs.
Requires OPENAI_BASE_URL, OPENAI_API_KEY, LLM_MODEL_NAME, EMBEDDER_MODEL_NAME.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from ragu import (
    ArtifactsExtractorLLM,
    BuilderArguments,
    KnowledgeGraph,
    LocalSearchEngine,
    Settings,
    StorageArguments,
)
from ragu.chunker import SimpleChunker
from ragu.common.prompts import ICLConfig
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI
from ragu.utils.ragu_utils import read_text_from_files

LANGUAGE = "russian"
DATA_DIR = Path("data")
STORAGE_FOLDER = Path("ragu_working_dir/demo")
CACHE_FOLDER = Path("cache")

QUESTIONS = [
    "<question 1>",
    "<question 2>",
]


async def main(do_build: bool) -> None:
    # Global settings come first: everything below reads them at construction time.
    # The storage folder is set explicitly because its default carries a timestamp,
    # which would send every run to a fresh, empty index.
    Settings.language = LANGUAGE
    Settings.storage_folder = STORAGE_FOLDER
    Settings.cache_path = CACHE_FOLDER

    client = CachedAsyncOpenAI(
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
        rate_max_simultaneous=10,
        rate_max_per_minute=100,
    )
    llm = LLMOpenAI(client=client, model_name=os.environ["LLM_MODEL_NAME"])
    embedder = EmbedderOpenAI(client=client, model_name=os.environ["EMBEDDER_MODEL_NAME"])
    await embedder.initialize()

    graph = KnowledgeGraph(
        llm=llm,
        embedder=embedder,
        chunker=SimpleChunker(max_chunk_size=1000),
        artifact_extractor=ArtifactsExtractorLLM(
            llm=llm,
            embedder=embedder,
            icl_config=ICLConfig(enabled=True, num_examples=2, selection_strategy="hybrid"),
            do_validation=True,
        ),
        builder_settings=BuilderArguments(use_llm_summarization=True),
        storage_settings=StorageArguments(),
    )

    if do_build:
        documents = read_text_from_files(DATA_DIR, file_extensions={".txt", ".md"})
        print(f"Indexing {len(documents)} documents into {Settings.storage_folder}")
        await graph.build_from_docs(documents)

    engine = LocalSearchEngine(llm, graph, embedder)

    for question in QUESTIONS:
        answer = await engine.query(question)
        print(f"\nQ: {question}\nA: {answer.response}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true",
                        help="Index the corpus before querying.")
    asyncio.run(main(parser.parse_args().build))
