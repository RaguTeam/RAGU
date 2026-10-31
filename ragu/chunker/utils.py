import base64
import mimetypes
import re
from dataclasses import dataclass, field
from typing import NamedTuple
from pathlib import Path
from typing import Any

from ragu.utils.ragu_utils import compute_mdhash_id

DESCRIPTION_LIMIT = 1000
CONTEXT_SIZE = 500
CONTENT_LIMIT = 8000
FIRST_H1 = re.compile(r"^#[ \t]+(.+?)[ \t#]*$", re.MULTILINE)
HEADING = re.compile(r"^#{1,2}[ \t]+(.+?)[ \t#]*$", re.MULTILINE)
HTML_TABLE = re.compile(r"<table\b.*?</table>", re.DOTALL | re.IGNORECASE)
FORMULA = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
IMAGE_ALT = re.compile(r"\[?!\[([^\]]*)\]")
TABLE_MARKUP = re.compile(r"<[^>]+>|\||-{3,}|\*\*")
EXTRA_NEWLINES = re.compile(r"\n{3,}")
MINERU_OBJECT_TYPES = frozenset({"table", "equation", "code", "image", "chart"})


@dataclass(slots=True)
class Span:
    """
    An object's half-open source range, content and chunk metadata.

    :param start: Inclusive start offset in the source Markdown.
    :param end: Exclusive end offset in the source Markdown.
    :param chunk_cls: Chunk class used to represent the object.
    :param content: Extracted object content.
    :param metadata: Object-specific metadata.
    """

    start: int
    end: int
    chunk_cls: type
    content: str
    metadata: dict[str, str] = field(default_factory=dict)


def typed_id(chunk: Any) -> str:
    # NOTE: identical formulas (or any identical chunks of one class) within one document get the same id
    return compute_mdhash_id(type(chunk).__name__, chunk.doc_id, chunk.content, prefix="chunk-")


def join(*parts: str | None) -> str:
    return "\n\n".join(part for part in parts if part)


def context_header(metadata: dict[str, Any]) -> str:
    labels = (("document_title", "Document"), ("section", "Section"))
    return "\n".join(f"{label}: {metadata[key]}" for key, label in labels if metadata.get(key))


def shorten(text: str) -> str:
    return text if len(text) <= DESCRIPTION_LIMIT else text[:DESCRIPTION_LIMIT].rsplit(" ", 1)[0] + "…"


def table_text(table: str) -> str:
    return shorten(" ".join(TABLE_MARKUP.sub(" ", table).split()))


def document_title(markdown: str, structured_content: dict[str, Any] | None) -> str | None:
    title = (structured_content or {}).get("metadata", {}).get("document", {}).get("title")
    match = FIRST_H1.search(markdown)
    return title or (match.group(1).strip() if match else None)


def normalize(text: str) -> str:
    return " ".join(text.split())


def mineru_captions(structured_content: dict[str, Any] | None) -> dict[str, str]:
    """
    Map normalized MinerU object content and image paths to captions.

    :param structured_content: Optional MinerU structured content.
    :return: Captions keyed by normalized object content or image path.
    """
    captions = {}
    for page in (structured_content or {}).get("pages", []):
        for block in page.get("blocks", []):
            caption = " ".join(item.get("content", "").strip() for item in block.get("captions") or []).strip()
            if block.get("type") in MINERU_OBJECT_TYPES and caption:
                for key in (block.get("image_source"), block.get("content")):
                    if key:
                        captions[normalize(key)] = caption
    return captions


def image_caption(text: str, start: int) -> dict[str, str]:
    match = IMAGE_ALT.match(text, start)
    return {"caption": match.group(1).strip()} if match and match.group(1).strip() else {}


def resolve(spans: list[Span]) -> list[Span]:
    """
    Keep non-overlapping ranges, preferring earlier starts and longer ties.

    :param spans: Candidate object ranges.
    :return: Non-overlapping ranges sorted by source position.
    """
    resolved: list[Span] = []
    for span in sorted(spans, key=lambda item: (item.start, -item.end)):
        if not resolved or span.start >= resolved[-1].end:
            resolved.append(span)
    return resolved


def mask(text: str, spans: list[Span]) -> str:
    for span in spans:
        text = text[:span.start] + " " * (span.end - span.start) + text[span.end:]
    return text


def clean(text: str) -> str:
    return EXTRA_NEWLINES.sub("\n\n", text).strip()


def description_request(chunk: Any, context: str, folder: str | None) -> str | list[dict[str, Any]]:
    """
    Build an LLM request, attaching only images inside the document folder.

    :param chunk: Object chunk to describe.
    :param context: Surrounding text used to interpret the object.
    :param folder: Document folder for local images, or ``None`` for text-only requests.
    :return: Plain text or a list of text and image message parts.
    :raises ValueError: If an image path resolves outside the document folder.
    """
    fields = (
        ("Object type", type(chunk).__name__.removesuffix("Chunk").lower()),
        ("Document", chunk.metadata.get("document_title")),
        ("Section", chunk.metadata.get("section")),
        ("Caption", chunk.metadata.get("caption")),
        ("Surrounding text", context),
        ("Content", chunk.content[:CONTENT_LIMIT]),
    )
    text = "\n\n".join(f"{label}: {value}" for label, value in fields if value)
    if not folder or type(chunk).__name__ != "ImageChunk":
        return text
    root = Path(folder).resolve()
    image = (root / chunk.content).resolve()
    if not image.is_relative_to(root):
        raise ValueError(f"Image path must stay within the document folder: {chunk.content}")
    if not image.is_file():
        return text
    mime = mimetypes.guess_type(image.name)[0] or "image/jpeg"
    url = f"data:{mime};base64,{base64.b64encode(image.read_bytes()).decode()}"
    return [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": url}}]


class Section(NamedTuple):
    """
    A heading and the source ranges of its heading and body.

    :param title: Section heading, or ``None`` for text before the first heading.
    :param start: Inclusive section start offset.
    :param body_start: Start offset of the text after the heading.
    :param end: Exclusive section end offset.
    """

    title: str | None
    start: int
    body_start: int
    end: int


def sections(masked: str) -> list[Section]:
    """
    Locate H1/H2 sections after object contents have been masked.

    :param masked: Markdown with object ranges replaced by spaces.
    :return: Section titles and source ranges, including the initial preamble.
    """
    headings = list(HEADING.finditer(masked))
    starts = [match.start() for match in headings] + [len(masked)]
    return [Section(None, 0, 0, starts[0])] + [
        Section(match.group(1).strip("*_ ") or None, match.start(), match.end(), end)
        for match, end in zip(headings, starts[1:], strict=True)
    ]


def surrounding_text(text: str, span: Span, section: Section) -> str:
    before = text[max(section.body_start, span.start - CONTEXT_SIZE):span.start]
    after = text[span.end:min(section.end, span.end + CONTEXT_SIZE)]
    return clean(f"{before}\n\n{after}")
