from dataclasses import dataclass, field

from ragu.utils.ragu_utils import compute_mdhash_id


@dataclass(slots=True)
class Chunk:
    id: str=field(init=False)
    content: str
    chunk_order_idx: int
    doc_id: str
    num_tokens: int | None = None

    def __post_init__(self) -> None:
        self.id = compute_mdhash_id(self.content, prefix="chunk-")

