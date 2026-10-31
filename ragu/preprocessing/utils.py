from collections.abc import Collection
from pathlib import Path

from mineru.parser import MinerUApiParser
from mineru.parser.writer import FileBasedDataWriter
from tqdm import tqdm

from ragu.common.logger import logger

MINERU_SUPPORTED_EXTENSIONS = frozenset({
    ".pdf",
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".jp2",
    ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".rtf",
    ".odt", ".ods", ".odp", ".epub", ".ofd",
    ".html", ".htm", ".shtml", ".mhtml", ".mht",
    ".csv", ".tsv",
})


def parse_from_folder(
    client: MinerUApiParser,
    folder: str | Path,
    result_folder: str | Path,
    file_extensions: Collection[str] | None = None,
) -> None:
    
    data = Path(folder)
    result = Path(result_folder)
    extensions = (
        MINERU_SUPPORTED_EXTENSIONS
        if file_extensions is None
        else {"." + ext.lower().lstrip(".") for ext in file_extensions}
    )
    files = [f for f in sorted(data.rglob("*")) if f.is_file() and f.suffix.lower() in extensions]

    for f in tqdm(files):
        out = result / f.relative_to(data).parent / f.stem
        try:
            out.mkdir(parents=True, exist_ok=True)
            client.parse(str(f)).save(FileBasedDataWriter(str(out)))
        except Exception as e:
            logger.error(f"Failed to parse {f}: {e}")