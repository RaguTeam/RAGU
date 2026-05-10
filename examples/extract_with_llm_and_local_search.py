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
    # Configure working directory and language
    Settings.storage_folder = "ragu_working_dir/example_knowledge_graph"
    Settings.language = "russian"

    # Load documents
    docs = read_text_from_files("examples/data/ru")

    # Initialize chunker
    chunker = SimpleChunker(max_chunk_size=1000)

    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    llm_model_name = os.getenv("LLM_MODEL_NAME")
    embedder_model_name = os.getenv("EMBEDDER_MODEL_NAME")

    # Set up shared OpenAI client
    client = CachedAsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
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
        language="russian",
    )

    # Set up artifact extractor with ICL
    artifact_extractor = ArtifactsExtractorLLM(
        llm=llm,
        embedder=embedder,
        icl_config=icl_config,
        do_validation=False,
    )

    # Configure graph builder
    builder_settings = BuilderArguments(
        use_llm_summarization=True,
        vectorize_chunks=True,
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
        tokenizer_model="gpt-4o-mini",
    )

    # Run local search
    questions = [
        "Кто написал гимн Норвегии?",
        "Шум, издаваемый ЭТИМИ ПАУКООБРАЗНЫМИ, слышен за пять километров. Отсюда и их название.",
        "Как переводится роман 'Ка́мо гряде́ши, Го́споди?'"
    ]

    for question in questions:
        print(await search_engine.a_query(question))

if __name__ == "__main__":
    asyncio.run(main())
