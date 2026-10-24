from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from chonkie import BaseChunker as ChonkieChunker
from chonkie import MarkdownChef, MarkdownDocument, RecursiveChunker, SemanticChunker, SentenceTransformerEmbeddings

from ragu.chunker.base_chunker import BaseChunker
from ragu.chunker.types import Chunk, CodeChunk, Document, FormulaChunk, ImageChunk, ObjectChunk, TableChunk, TextChunk
from ragu.chunker.utils import (
    DESCRIPTION_PROMPT,
    FORMULA,
    HTML_TABLE,
    Section,
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
    surrounding_text,
)
from ragu.models.llm import LLM


@dataclass(slots=True)
class _Item:
    position: int
    chunk: Chunk
    context: str = ""


class MarkdownChunker(BaseChunker):
    def __init__(self, text_chunker: ChonkieChunker, llm: LLM | None = None) -> None:
        super().__init__()
        self.text_chunker = text_chunker
        self.llm = llm
        self.chef = MarkdownChef()

    def split(self, documents: str | Document | Sequence[str | Document]) -> list[Chunk]:
        return [item.chunk for document in _documents(documents) for item in self._split(document)]

    async def a_split(self, documents: str | Document | Sequence[str | Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in _documents(documents):
            items = self._split(document)
            if self.llm:
                await self._describe(document, items)
            chunks.extend(item.chunk for item in items)
        return chunks

    async def _describe(self, document: Document, items: list[_Item]) -> None:
        assert self.llm is not None
        targets = [item for item in items if isinstance(item.chunk, ObjectChunk)]
        if not targets:
            return

        conversations: list[Any] = [
            [{"role": "system", "content": DESCRIPTION_PROMPT},
             {"role": "user", "content": description_request(item.chunk, item.context, document.folder)}]
            for item in targets
        ]
        results = await self.llm.batch_chat_completion(
            conversations, continue_on_error=True, desc="Describing tables, formulas, code and images"
        )
        for item, result in zip(targets, results, strict=True):
            if result and result.strip():
                item.chunk.metadata["description"] = result.strip()

    def _split(self, document: Document) -> list[_Item]:
        spans = self._find_objects(document.content)
        captions = mineru_captions(document.structured_content)

        items = [
            item
            for section in sections(mask(document.content, spans))
            for item in self._section_items(document, section, spans, captions)
        ]
        items.sort(key=lambda item: item.position)
        for index, item in enumerate(items):
            item.chunk.chunk_order_idx = index
        return items

    def _section_items(
        self, document: Document, section: Section, spans: list[Span], captions: dict[str, str]
    ) -> list[_Item]:
        metadata = {"document_title": document.title, "section": section.title}
        metadata = {key: value for key, value in metadata.items() if value}
        inner = [span for span in spans if section.start <= span.start < section.end]
        return [
            *self._text_items(document, section, inner, metadata),
            *(self._object_item(document, section, span, metadata, captions) for span in inner),
        ]

    def _text_items(
        self, document: Document, section: Section, spans: list[Span], metadata: dict[str, Any]
    ) -> list[_Item]:
        text, bounds = cut(document.content, section.body_start, section.end, spans)
        if not text.strip():
            return []

        items = []
        for piece in self.text_chunker.chunk(text):
            if content := clean(piece.text):
                offset = piece.start_index + len(piece.text) - len(piece.text.lstrip())
                chunk = TextChunk(content, 0, document.id, metadata=dict(metadata))
                items.append(_Item(original_position(bounds, offset), chunk))
        return items

    @staticmethod
    def _object_item(
        document: Document, section: Section, span: Span, metadata: dict[str, Any], captions: dict[str, str]
    ) -> _Item:
        chunk = span.chunk_cls(span.content, 0, document.id, metadata={**metadata, **span.metadata})
        if caption := captions.get(normalize(span.content)):
            chunk.metadata["caption"] = caption
        chunk.describe()
        return _Item(span.start, chunk, surrounding_text(document.content, span, section))

    def _find_objects(self, text: str) -> list[Span]:
        # Code goes first so that "#" and "$$" inside it are ignored; formulas are searched with code and tables masked.
        parsed = self.chef.parse(text + "\n")
        code = resolve(_code_spans(parsed, text))
        blocks = resolve(code + _table_spans(parsed, text, code))
        return resolve(blocks + _formula_spans(text, blocks) + _image_spans(parsed, text))


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


def _code_spans(parsed: MarkdownDocument, text: str) -> list[Span]:
    return [
        Span(block.start_index, min(block.end_index, len(text)), CodeChunk, block.content,
             {"language": block.language} if block.language else {})
        for block in parsed.code
    ]


def _table_spans(parsed: MarkdownDocument, text: str, code: list[Span]) -> list[Span]:
    pipe = [
        Span(table.start_index, min(table.end_index, len(text)), TableChunk, table.content.strip())
        for table in parsed.tables
    ]
    html = [
        Span(match.start(), match.end(), TableChunk, match.group(0))
        for match in HTML_TABLE.finditer(mask(text, code))
    ]
    return pipe + html


def _formula_spans(text: str, blocks: list[Span]) -> list[Span]:
    return [
        Span(match.start(), match.end(), FormulaChunk, match.group(1).strip())
        for match in FORMULA.finditer(mask(text, blocks))
    ]


def _image_spans(parsed: MarkdownDocument, text: str) -> list[Span]:
    return [
        Span(image.start_index, min(image.end_index, len(text)), ImageChunk, image.content,
             image_caption(text, image.start_index))
        for image in parsed.images
    ]
