"""
Example: interactive question answering over an existing pre-built RAGU index.

Unlike the build-oriented examples, this script never constructs a graph: it
attaches to index files already on disk (``knowledge_graph.gml``, ``kv_*.json``,
``vdb_*.json``), so every run is cheap and reproducible. It is intended for
debugging retrieval quality — e.g. comparing scores for the same question asked
in different languages — and for quick engine-configuration experiments.

Engine selection rules:

- one engine in ``--engine`` → that engine is used directly;
- two or more → they are wrapped into ``MixSearchEngine`` automatically;
- ``--query-plan`` additionally wraps the result into ``QueryPlanEngine``
  (decomposition runs only in answer-generation mode, not with ``--no-llm``).

By default (``--language auto``) the answer language is detected per question
with ``lingua-language-detector`` (Russian/English), so the answer follows the
language of the question. Pass an explicit value (``english``, ``russian``, ...)
to fix the language.

Usage
-----
1. Serve a local embedder, e.g. with vLLM::

        vllm serve Alibaba-NLP/gte-multilingual-base \
            --hf-overrides '{"architectures": ["GteNewModel"]}' \
            --runner pooling --trust-remote-code --max-model-len 2048 \
            --host 127.0.0.1 --port 8001

2. Ask a question (answer generation, needs an LM endpoint)::

        export OPENAI_BASE_URL="https://..." OPENAI_API_KEY="sk-..." LLM_MODEL_NAME="gpt-4o-mini"
        python examples/generate_answer.py \
            --index /path/to/index --engine local naive "What is skin cancer?"

3. Retrieval-only debugging (no LLM needed; scores and contexts are logged)::

        python examples/generate_answer.py \
            --index /path/to/index --engine local --no-llm "Что такое рак кожи?"

4. Interactive REPL: omit the question argument.

All endpoint defaults can be overridden via CLI flags or the environment
variables ``EMBEDDER_BASE_URL`` / ``LOCAL_EMBEDDER_URL``, ``EMBEDDER_MODEL_NAME``,
``EMBEDDER_API_KEY``, ``OPENAI_BASE_URL`` / ``LLM_BASE_URL``,
``LLM_MODEL_NAME`` / ``OPENAI_MODEL``, ``OPENAI_API_KEY`` / ``LLM_API_KEY``.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import Any

from lingua import Language, LanguageDetectorBuilder

from ragu import (
    KnowledgeGraph,
    LocalSearchEngine,
    MixSearchEngine,
    NaiveSearchEngine,
    QueryPlanEngine,
    Settings,
    StorageArguments,
)
from ragu.common.logger import logger
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI
from ragu.search_engine.base_engine import EngineParams, SearchEngineRetrieve
from ragu.search_engine.local_search import LocalParams, LocalSearchRetrieve
from ragu.search_engine.mix_search import MixQueryParams, MixSearchRetrieve
from ragu.search_engine.naive_search import NaiveSearchParams, NaiveSearchRetrieve

INDEX_VDB_FILES = ("vdb_entity.json", "vdb_chunk.json", "vdb_relation.json")
PREVIEW_LENGTH = 110


def _env_first(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


class _SearchOnlyLLM:
    """LLM stub that allows engine construction but forbids generation."""

    async def chat_completion(self, *_args: Any, **_kwargs: Any) -> str:
        raise RuntimeError("LLM generation is disabled: restart without --no-llm.")

    async def batch_chat_completion(self, *_args: Any, **_kwargs: Any) -> list:
        raise RuntimeError("LLM generation is disabled: restart without --no-llm.")


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Ask questions against an existing RAGU index with detailed search logging."
    )
    parser.add_argument("--index", type=Path, required=True, help="Folder with a pre-built RAGU index.")
    parser.add_argument(
        "--engine",
        nargs="+",
        choices=["local", "naive"],
        default=["local"],
        help="One engine to use, or several to wrap them into MixSearchEngine.",
    )
    parser.add_argument(
        "--query-plan",
        action="store_true",
        help="Wrap the engine into QueryPlanEngine (applies to answer generation only).",
    )
    parser.add_argument(
        "--llm",
        action=BooleanOptionalAction,
        default=True,
        help="Generate answers with an LLM (--no-llm logs retrieval only, no LM endpoint needed).",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Number of retrieved items per query.")
    parser.add_argument(
        "--language",
        default="auto",
        help="Answer language: 'auto' detects the question's language (ru/en, via lingua); "
        "any other value (english, russian, ...) is fixed and fed into RAGU prompts as-is.",
    )
    parser.add_argument(
        "--embed-base-url",
        default=_env_first("EMBEDDER_BASE_URL", "LOCAL_EMBEDDER_URL", default="http://127.0.0.1:8001/v1"),
        help="OpenAI-compatible embeddings endpoint.",
    )
    parser.add_argument(
        "--embed-model",
        default=_env_first("EMBEDDER_MODEL_NAME", default="Alibaba-NLP/gte-multilingual-base"),
        help="Embedding model name as served by the endpoint.",
    )
    parser.add_argument(
        "--embed-api-key",
        default=_env_first("EMBEDDER_API_KEY", default="unused"),
        help="API key for the embeddings endpoint.",
    )
    parser.add_argument(
        "--embed-token-limit",
        type=int,
        default=2000,
        help="Queries are truncated to this many tokens before embedding.",
    )
    parser.add_argument(
        "--llm-base-url",
        default=_env_first("OPENAI_BASE_URL", "LLM_BASE_URL"),
        help="OpenAI-compatible chat endpoint.",
    )
    parser.add_argument(
        "--llm-model",
        default=_env_first("LLM_MODEL_NAME", "OPENAI_MODEL"),
        help="Chat model name.",
    )
    parser.add_argument(
        "--llm-api-key",
        default=_env_first("OPENAI_API_KEY", "LLM_API_KEY"),
        help="API key for the chat endpoint.",
    )
    parser.add_argument(
        "--show-context",
        action=BooleanOptionalAction,
        default=True,
        help="Print the exact context that is fed to the LLM.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity for both this script and RAGU internals.",
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="One-shot question; omit it to enter the interactive REPL.",
    )
    return parser


def _read_index_embedding_dim(index_path: Path) -> int | None:
    for filename in INDEX_VDB_FILES:
        vdb_path = index_path / filename
        if not vdb_path.exists():
            continue
        with vdb_path.open(encoding="utf-8") as file:
            dim = json.load(file).get("embedding_dim")
        if isinstance(dim, int) and dim > 0:
            return dim
    return None


def _build_embedder(args: ArgumentParser) -> EmbedderOpenAI:
    client = CachedAsyncOpenAI(
        base_url=args.embed_base_url,
        api_key=args.embed_api_key,
        rate_max_simultaneous=20,
        rate_max_per_minute=500,
        embed_timeout=60.0,
    )
    common = {
        "client": client,
        "model_name": args.embed_model,
        "batch_size": 32,
        "max_concurrent_batches": 2,
        "embedder_token_limit": args.embed_token_limit,
    }
    try:
        return EmbedderOpenAI(
            **common,
            tokenizer_backend="local",
            tokenizer_name=args.embed_model,
        )
    except ValueError as error:
        logger.warning(
            "Local tokenizer for '{}' is unavailable ({}); "
            "falling back to the tiktoken backend.",
            args.embed_model,
            error,
        )
    return EmbedderOpenAI(
        **common,
        tokenizer_backend="tiktoken",
        tokenizer_name=Settings.tokenizer_embedder_name,
    )


def _build_llm(args: ArgumentParser) -> Any:
    if not args.llm:
        return _SearchOnlyLLM()
    if not args.llm_model:
        raise SystemExit(
            "Answer generation requires an LLM: set --llm-model / --llm-base-url "
            "or the LLM_MODEL_NAME / OPENAI_BASE_URL environment variables "
            "(or run with --no-llm for retrieval-only debugging)."
        )
    client = CachedAsyncOpenAI(
        base_url=args.llm_base_url,
        api_key=args.llm_api_key or "unused",
        rate_max_simultaneous=10,
        rate_max_per_minute=100,
    )
    return LLMOpenAI(client=client, model_name=args.llm_model)


def _build_engine(args: ArgumentParser, llm: Any, knowledge_graph: KnowledgeGraph, embedder: EmbedderOpenAI) -> Any:
    # Placeholder until ask_once resolves the real per-question language.
    language = "english" if args.language == "auto" else args.language
    child_params: list[EngineParams | None] = []
    engines: list[Any] = []

    def _add_local() -> None:
        engines.append(
            LocalSearchEngine(
                llm=llm,
                knowledge_graph=knowledge_graph,
                embedder=embedder,
                language=language,
            )
        )
        child_params.append(LocalParams(top_k=args.top_k))

    def _add_naive() -> None:
        engines.append(
            NaiveSearchEngine(
                llm=llm,
                knowledge_graph=knowledge_graph,
                embedder=embedder,
                language=language,
            )
        )
        child_params.append(NaiveSearchParams(top_k=args.top_k))

    for name in args.engine:
        {"local": _add_local, "naive": _add_naive}[name]()

    if len(engines) == 1:
        engine = engines[0]
        logger.info("Engine: {}", type(engine).__name__)
    else:
        engine = MixSearchEngine(
            llm=llm,
            engines=engines,
            engine_params=child_params,
            allow_partial_failures=True,
            language=language,
        )
        logger.info(
            "Engine: MixSearchEngine over [{}]",
            ", ".join(type(child).__name__ for child in engines),
        )

    if args.query_plan:
        if not args.llm:
            raise SystemExit("--query-plan decomposes questions with an LLM and cannot run with --no-llm.")
        # QueryPlanEngine takes no language argument; answer language is applied
        # to its child engines via _apply_language below.
        engine = QueryPlanEngine(engine=engine)
        logger.info("Wrapped into QueryPlanEngine")
    return engine


_language_detector = None
_LANGUAGE_NAMES = {Language.RUSSIAN: "russian", Language.ENGLISH: "english"}


def _detect_language(question: str) -> str:
    """Detect the question language (ru/en) with a lazily built lingua detector."""
    global _language_detector
    if _language_detector is None:
        _language_detector = LanguageDetectorBuilder.from_languages(
            Language.RUSSIAN, Language.ENGLISH
        ).build()
    detected = _language_detector.detect_language_of(question)
    if detected is None:
        logger.warning("Could not detect the question language; defaulting to english.")
        return "english"
    return _LANGUAGE_NAMES[detected]


def _apply_language(engine: Any, language: str) -> None:
    """Set the answer language on every engine in the tree that supports it."""
    if hasattr(engine, "language"):
        engine.language = language
    for child in getattr(engine, "engines", []) or []:
        _apply_language(child, language)
    inner = getattr(engine, "engine", None)
    if inner is not None and inner is not engine:
        _apply_language(inner, language)
    Settings.language = language


def _resolve_language(args: ArgumentParser, question: str) -> str:
    if args.language != "auto":
        return args.language
    return _detect_language(question)


def _query_params(args: ArgumentParser) -> EngineParams:
    if len(args.engine) > 1:
        return MixQueryParams()
    if "local" in args.engine:
        return LocalParams(top_k=args.top_k)
    return NaiveSearchParams(top_k=args.top_k)


def _preview(text: str, limit: int = PREVIEW_LENGTH) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _describe_retrieval(retrieval: SearchEngineRetrieve, top_n: int = 5) -> None:
    if isinstance(retrieval, MixSearchRetrieve):
        children = retrieval.result.results
        logger.info("MixSearchEngine: {} child result(s)", len(children))
        for child in children:
            _describe_retrieval(child, top_n)
        return

    if isinstance(retrieval, LocalSearchRetrieve):
        result = retrieval.result
        logger.info(
            "LocalSearchEngine: entities={}, relations={}, summaries={}, chunks={}, documents={}",
            len(result.entities),
            len(result.relations),
            len(result.summaries),
            len(result.chunks),
            len(result.documents_id),
        )
        scores = {
            str(item.get("id")): item.get("relevance_score")
            for item in retrieval.metrics.get("entities", [])
            if isinstance(item, dict)
        }
        for entity in result.entities[:top_n]:
            logger.info(
                "  entity  score={:.4f}  {} [{}]",
                float(scores.get(entity.id) or 0.0),
                entity.entity_name,
                entity.entity_type,
            )
        for chunk in result.chunks[:top_n]:
            logger.info("  chunk   {} | {}", chunk.id, _preview(chunk.content))
        return

    if isinstance(retrieval, NaiveSearchRetrieve):
        result = retrieval.result
        logger.info("NaiveSearchEngine: chunks={}, documents={}", len(result.chunks), len(result.documents_id))
        for chunk, score in zip(result.chunks, result.scores):
            logger.info("  chunk   score={:.4f}  {} | {}", float(score), chunk.id, _preview(chunk.content))
        return

    logger.warning("Unhandled retrieval type: {}", type(retrieval).__name__)


def _check_mix_engines(engine: Any, retrieval: SearchEngineRetrieve) -> None:
    engines = getattr(engine, "engines", None)
    if not engines or not isinstance(retrieval, MixSearchRetrieve):
        return
    used = len(retrieval.result.results)
    if used < len(engines):
        logger.warning(
            "MixSearchEngine used only {} of {} child engines; the rest failed and were dropped.",
            used,
            len(engines),
        )


async def ask_once(engine: Any, args: ArgumentParser, question: str) -> None:
    language = _resolve_language(args, question)
    source = "auto-detected from question" if args.language == "auto" else "fixed via --language"
    logger.info("Answer language: {} ({})", language, source)
    _apply_language(engine, language)

    print("=" * 80)
    print(f"Question: {question}")
    started = time.perf_counter()
    try:
        if args.llm:
            response = await engine.query(question, params=_query_params(args))
            retrieval = response.retrieval
            answer = str(response)
        else:
            retrieval = await engine.search(question, params=_query_params(args))
            answer = "[search-only mode: no LLM answer was generated]"
    except Exception as error:
        logger.exception("Query failed: {}: {}", type(error).__name__, error)
        return
    elapsed = time.perf_counter() - started

    print("-" * 80)
    print(f"Answer: {answer}")
    print("-" * 80)
    logger.info("Query finished in {:.0f} ms", elapsed * 1000)
    _check_mix_engines(engine, retrieval)
    _describe_retrieval(retrieval)
    if args.show_context:
        print("-" * 80)
        print("Context fed to the LLM:")
        print(retrieval.to_text())


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logger.remove()
    logger.add(
        sys.stdout,
        level=args.log_level,
        colorize=True,
        format="<cyan>{time:HH:mm:ss}</cyan> | <level>{level: <8}</level> | <level>{message}</level>",
    )

    index_path = args.index.expanduser().resolve()
    if not (index_path / "knowledge_graph.gml").exists():
        raise SystemExit(f"'{index_path}' does not look like a RAGU index (no knowledge_graph.gml).")

    Settings.storage_folder = str(index_path)
    Settings.embedder_token_limit = args.embed_token_limit

    llm = _build_llm(args)
    embedder = _build_embedder(args)
    await embedder.initialize()

    index_dim = _read_index_embedding_dim(index_path)
    logger.info("Index: {}", index_path)
    logger.info("Embedder '{}' at '{}': dim={}", args.embed_model, args.embed_base_url, embedder.dim)
    if index_dim is None:
        logger.warning("No embedding_dim found in index files; skipping the dimension check.")
    elif index_dim != embedder.dim:
        raise SystemExit(
            f"Embedding dimension mismatch: index stores {index_dim}-d vectors, "
            f"but '{args.embed_model}' returns {embedder.dim}-d. "
            "Use the model the index was built with."
        )
    else:
        logger.info("Dimension check passed: index {}-d == embedder {}-d", index_dim, embedder.dim)

    # Placeholder until ask_once resolves the real per-question language.
    initial_language = "english" if args.language == "auto" else args.language
    knowledge_graph = KnowledgeGraph(
        llm=llm,
        embedder=embedder,
        storage_settings=StorageArguments(),
        language=initial_language,
    )
    engine = _build_engine(args, llm, knowledge_graph, embedder)
    language_log = "auto (per-question detection)" if args.language == "auto" else args.language
    logger.info(
        "top_k={}, language={}, query_plan={}", args.top_k, language_log, args.query_plan
    )

    if args.question:
        await ask_once(engine, args, " ".join(args.question))
        return

    print("Enter questions one per line; an empty line or Ctrl-D/Ctrl-C exits.")
    while True:
        try:
            question = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        await ask_once(engine, args, question)


if __name__ == "__main__":
    asyncio.run(main())
