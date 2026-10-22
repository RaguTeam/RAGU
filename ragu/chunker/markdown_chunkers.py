from collections.abc import Sequence
from typing import Any

from chonkie import BaseChunker as ChonkieChunker
from chonkie import MarkdownChef, RecursiveChunker, SemanticChunker, SentenceTransformerEmbeddings

from ragu.chunker.base_chunker import BaseChunker
from ragu.chunker.types import Chunk, CodeChunk, Document, FormulaChunk, ImageChunk, ObjectChunk, TableChunk, TextChunk
from ragu.chunker.utils import (
    CONTEXT_SIZE,
    DESCRIPTION_PROMPT,
    FORMULA,
    HTML_TABLE,
    Span,
    clean,
    cut,
    description_request,
    image_caption,
    mask,
    mineru_captions,
    normalize,
    original_position,
    resolve,
    sections,
)
from ragu.models.llm import LLM


class MarkdownChunker(BaseChunker):
    def __init__(self, text_chunker: ChonkieChunker, llm: LLM | None = None) -> None:
        super().__init__()
        self.text_chunker = text_chunker
        self.llm = llm
        self.chef = MarkdownChef()

    def split(self, documents: str | Document | Sequence[str | Document]) -> list[Chunk]:
        return [chunk for document in _documents(documents) for chunk, _ in self._split(document)]

    async def a_split(self, documents: str | Document | Sequence[str | Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in _documents(documents):
            pairs = self._split(document)
            if self.llm:
                await self._describe(document, pairs)
            chunks.extend(chunk for chunk, _ in pairs)
        return chunks

    async def _describe(self, document: Document, pairs: list[tuple[Chunk, str]]) -> None:
        assert self.llm is not None
        targets = [(chunk, context) for chunk, context in pairs if isinstance(chunk, ObjectChunk)]

        if not targets:
            return
        
        conversations: list[Any] = [
            [{"role": "system", "content": DESCRIPTION_PROMPT},
             {"role": "user", "content": description_request(chunk, context, document.folder)}]
            for chunk, context in targets
        ]
        results = await self.llm.batch_chat_completion(
            conversations, continue_on_error=True, desc="Describing tables, formulas, code and images"
        )

        for (chunk, _), result in zip(targets, results, strict=True):
            if result and result.strip():
                chunk.metadata["description"] = result.strip()

    def _split(self, document: Document) -> list[tuple[Chunk, str]]:
        text = document.content
        spans = self._find_objects(text)
        captions = mineru_captions(document.structured_content)
        items: list[tuple[int, Chunk, str]] = []

        for section, start, body_start, end in sections(mask(text, spans)):
            metadata = {"document_title": document.title, "section": section}
            metadata = {key: value for key, value in metadata.items() if value}
            inner = [span for span in spans if start <= span.start < end]
            section_text, bounds = cut(text, body_start, end, inner)

            for piece in self.text_chunker.chunk(section_text) if section_text.strip() else []:
                if content := clean(piece.text):
                    chunk = TextChunk(content, 0, document.id, metadata=dict(metadata))
                    offset = piece.start_index + len(piece.text) - len(piece.text.lstrip())
                    items.append((original_position(bounds, offset), chunk, ""))

            for span in inner:
                obj = span.chunk_cls(span.content, 0, document.id, metadata={**metadata, **span.metadata})
                if caption := captions.get(normalize(span.content)):
                    obj.metadata["caption"] = caption

                obj.describe()
                context = clean(text[max(body_start, span.start - CONTEXT_SIZE):span.start]
                                + "\n\n" + text[span.end:min(end, span.end + CONTEXT_SIZE)])
                items.append((span.start, obj, context))

        items.sort(key=lambda item: item[0])
        for index, item in enumerate(items):
            item[1].chunk_order_idx = index

        return [(item[1], item[2]) for item in items]

    def _find_objects(self, text: str) -> list[Span]:
        parsed = self.chef.parse(text + "\n")
        size = len(text)
        code = resolve([
            Span(block.start_index, min(block.end_index, size), CodeChunk, block.content,
                 {"language": block.language} if block.language else {})
            for block in parsed.code
        ])
        blocks = resolve(
            code
            + [Span(table.start_index, min(table.end_index, size), TableChunk, table.content.strip())
               for table in parsed.tables]
            + [Span(match.start(), match.end(), TableChunk, match.group(0))
               for match in HTML_TABLE.finditer(mask(text, code))]
        )
        return resolve(
            blocks
            + [Span(match.start(), match.end(), FormulaChunk, match.group(1).strip())
               for match in FORMULA.finditer(mask(text, blocks))]
            + [Span(image.start_index, min(image.end_index, size), ImageChunk, image.content,
                    image_caption(text, image.start_index))
               for image in parsed.images]
        )


class RecursiveMarkdownChunker(MarkdownChunker):
    def __init__(
        self, max_chunk_size: int = 1000, tokenizer: Any = "character", llm: LLM | None = None, **kwargs: Any
    ) -> None:
        super().__init__(RecursiveChunker(tokenizer=tokenizer, chunk_size=max_chunk_size, **kwargs), llm)


class SemanticMarkdownChunker(MarkdownChunker):
    def __init__(
        self,
        model_name: str,
        max_chunk_size: int = 512,
        threshold: float = 0.8,
        device: str | None = None,
        llm: LLM | None = None,
        **kwargs: Any,
    ) -> None:
        embeddings = SentenceTransformerEmbeddings(model_name, device=device)
        super().__init__(SemanticChunker(embedding_model=embeddings, chunk_size=max_chunk_size,
                                         threshold=threshold, **kwargs), llm)


def _documents(documents: str | Document | Sequence[str | Document]) -> list[Document]:
    if isinstance(documents, (str, Document)):
        documents = [documents]
    return [document if isinstance(document, Document) else Document(document) for document in documents]
