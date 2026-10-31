import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragu.chunker.utils import context_header, document_title, join, shorten, table_text, typed_id
from ragu.utils.ragu_utils import compute_mdhash_id


@dataclass(slots=True)
class Document:
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
        structure = Path(folder) / "structured_content.json"
        return cls(
            (Path(folder) / "markdown.md").read_text(encoding="utf-8"),
            json.loads(structure.read_text(encoding="utf-8")) if structure.is_file() else None,
            folder=str(folder),
        )


@dataclass(slots=True)
class Chunk:
    id: str = field(init=False)
    content: str
    chunk_order_idx: int
    doc_id: str
    num_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = compute_mdhash_id(self.content, prefix="chunk-")

    def to_embed(self) -> str:
        return self.content

    def to_llm(self) -> str:
        return self.content


@dataclass(slots=True)
class TextChunk(Chunk):
    def __post_init__(self) -> None:
        self.id = typed_id(self)

    def to_embed(self) -> str:
        return join(self.metadata.get("section"), self.content)

    def to_llm(self) -> str:
        return join(context_header(self.metadata), self.content)


@dataclass(slots=True)
class ObjectChunk(Chunk):
    def __post_init__(self) -> None:
        self.id = typed_id(self)

    def describe(self) -> str:
        # NOTE: placeholder; MarkdownChunker.a_split replaces it with an LLM description
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
    def summary(self) -> str:
        return table_text(self.content)


@dataclass(slots=True)
class FormulaChunk(ObjectChunk):
    def body(self) -> str:
        return f"$$\n{self.content}\n$$"


@dataclass(slots=True)
class CodeChunk(ObjectChunk):
    def summary(self) -> str:
        return shorten(self.content)

    def body(self) -> str:
        return f"```{self.metadata.get('language', '')}\n{self.content}\n```"


@dataclass(slots=True)
class ImageChunk(ObjectChunk):
    def summary(self) -> str:
        return ""

    def body(self) -> str:
        description = self.metadata.get("description", "")
        return "" if description in (self.metadata.get("caption"), self.metadata.get("section")) else description
