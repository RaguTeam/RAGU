import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragu.chunker.utils import context_header, document_title, join, shorten, table_text, typed_id
from ragu.utils.ragu_utils import compute_mdhash_id


@dataclass(slots=True)
class Document:
    """
    Markdown content with optional MinerU structure and a folder for local images.

    :param content: Markdown content.
    :param structured_content: Optional MinerU structure containing metadata and captions.
    :param title: Document title; inferred from metadata or Markdown if omitted.
    :param id: Document ID; derived from content when set to ``"auto"``.
    :param folder: Base folder for local image attachments.
    """

    content: str
    structured_content: dict[str, Any] | None = None
    title: str | None = None
    id: str = "auto"
    folder: str | None = None

    def __post_init__(self) -> None:
        if self.id == "auto":
            self.id = compute_mdhash_id(self.content)
        if self.title is None:
            self.title = document_title(self.content, self.structured_content)

    @classmethod
    def from_mineru(cls, folder: str | Path) -> "Document":
        """
        Load MinerU Markdown and optional structured content from a result folder.

        :param folder: Folder containing ``markdown.md`` and optional ``structured_content.json``.
        :return: Document with content, metadata and the image base folder.
        """
        structure = Path(folder) / "structured_content.json"
        return cls(
            (Path(folder) / "markdown.md").read_text(encoding="utf-8"),
            json.loads(structure.read_text(encoding="utf-8")) if structure.is_file() else None,
            folder=str(folder),
        )


@dataclass(slots=True)
class Chunk:
    """
    A text fragment with a deterministic ID and source document reference.

    :param content: Chunk content.
    :param chunk_order_idx: Zero-based position within the source document.
    :param doc_id: Source document ID.
    :param num_tokens: Optional token count.
    :param metadata: Additional document, section or object metadata.
    """

    id: str = field(init=False)
    content: str
    chunk_order_idx: int
    doc_id: str
    num_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = compute_mdhash_id(self.content, prefix="chunk-")

    def to_embed(self) -> str:
        """
        Return the text representation used for vectorization.

        :return: Text to embed.
        """
        return self.content

    def to_llm(self) -> str:
        """
        Return the text representation used in answer context.

        :return: Text to include in the LLM context.
        """
        return self.content


@dataclass(slots=True)
class TextChunk(Chunk):
    """
    Plain text with section context for embedding and answer generation.
    """

    def __post_init__(self) -> None:
        self.id = typed_id(self)

    def to_embed(self) -> str:
        return join(self.metadata.get("section"), self.content)

    def to_llm(self) -> str:
        return join(context_header(self.metadata), self.content)


@dataclass(slots=True)
class ObjectChunk(Chunk):
    """
    A Markdown object with a local fallback or LLM-generated description.
    """

    def __post_init__(self) -> None:
        self.id = typed_id(self)

    def describe(self) -> str:
        """
        Store a fallback description that async chunking may replace with LLM output.

        :return: Description stored in the chunk metadata.
        """
        description = join(self.metadata.get("caption"), self.summary()) or self.metadata.get("section") or self.content
        self.metadata["description"] = description
        return description

    def summary(self) -> str:
        return self.content

    def body(self) -> str:
        return self.content

    def to_embed(self) -> str:
        return self.metadata.get("description") or self.content

    def to_llm(self) -> str:
        return join(context_header(self.metadata), self.metadata.get("caption"), self.body())


@dataclass(slots=True)
class TableChunk(ObjectChunk):
    """
    A Markdown or HTML table with a compact plain-text summary.
    """

    def summary(self) -> str:
        return table_text(self.content)


@dataclass(slots=True)
class FormulaChunk(ObjectChunk):
    """
    A display equation whose LaTeX delimiters are restored for the LLM.
    """

    def body(self) -> str:
        return f"$$\n{self.content}\n$$"


@dataclass(slots=True)
class CodeChunk(ObjectChunk):
    """
    A code block with its language and Markdown representation.
    """

    def summary(self) -> str:
        return shorten(self.content)

    def body(self) -> str:
        return f"```{self.metadata.get('language', '')}\n{self.content}\n```"


@dataclass(slots=True)
class ImageChunk(ObjectChunk):
    """
    An image path whose description supplies textual answer context.
    """

    def summary(self) -> str:
        return ""

    def body(self) -> str:
        description = self.metadata.get("description", "")
        return "" if description in (self.metadata.get("caption"), self.metadata.get("section")) else description
