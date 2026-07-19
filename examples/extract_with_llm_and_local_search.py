from argparse import ArgumentParser
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
    parser = ArgumentParser()
    parser.add_argument("--lang", dest="language", type=str, required=False, default="ru",
                        choices=["ru", "rus", "russian", "en", "eng", "english"],
                        help="The language for demonstration.")
    args = parser.parse_args()

    # Configure working directory and language
    Settings.language = "russian" if args.language.startswith("ru") else "english"
    Settings.storage_folder = "ragu_working_dir/example_knowledge_graph_" + Settings.language

    # Load documents
    docs = read_text_from_files("examples/data/ru" if args.language.startswith("ru") else "examples/data/en")

    # Initialize chunker
    chunker = SimpleChunker(max_chunk_size=1000)

    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    llm_model_name = os.getenv("LLM_MODEL_NAME")
    embedder_model_name = os.getenv("EMBEDDER_MODEL_NAME")

    # Set up shared OpenAI client with rate limiting
    client = CachedAsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        rate_max_simultaneous=10,
        rate_max_per_minute=100,
    )

    llm = LLMOpenAI(
        client=client,
        model_name=llm_model_name,
    )

    embedder = EmbedderOpenAI(
        client=client,
        model_name=embedder_model_name,
    )
    await embedder.initialize()

    # Configure in-context learning (optional, improves extraction quality)
    icl_config = ICLConfig(
        enabled=True,
        num_examples=2,
        selection_strategy="hybrid"
    )

    # Set up artifact extractor with ICL
    artifact_extractor = ArtifactsExtractorLLM(
        llm=llm,
        embedder=embedder,
        icl_config=icl_config,
        do_validation=True,
    )

    # Configure graph builder
    builder_settings = BuilderArguments(
        use_llm_summarization=True,
    )

    # Build knowledge graph
    knowledge_graph = KnowledgeGraph(
        llm=llm,
        embedder=embedder,
        chunker=chunker,
        artifact_extractor=artifact_extractor,
        builder_settings=builder_settings,
    )
    await knowledge_graph.build_from_docs(docs)

    # Set up search engine
    search_engine = LocalSearchEngine(
        llm,
        knowledge_graph,
        embedder,
    )

    # Run local search
    if args.language.startswith("ru"):
        questions = [
            "Кто написал гимн Норвегии?",
            "Шум, издаваемый ЭТИМИ ПАУКООБРАЗНЫМИ, слышен за пять километров. Отсюда и их название.",
            "Как переводится название романа 'Ка́мо гряде́ши, Го́споди?' на русский языке"
        ]

        for question in questions:
            print(f'\nВопрос: {question}')
            answer = await search_engine.a_query(question)
            print(f'Ответ: {answer.response}')
    else:
        questions = [
            "Where did the father of the creator of the C programming language work?",
            "What did the person who died on October 12, 2011, create in their lifetime?"
        ]

        for question in questions:
            print(f'\nQuestion: {question}')
            answer = await search_engine.a_query(question)
            print(f'Answer: {answer.response}')


if __name__ == "__main__":
    asyncio.run(main())
