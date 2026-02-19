import time
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional

import numpy as np

from ragu.utils.ragu_utils import compute_mdhash_id


@dataclass(slots=True)
class Embedding:
    """
    Representation of an embedding.

    :param id: Unique record identifier.
    :param vector: Embedding vector.
    :param metadata: Additional payload.
    """
    vector: List[float] | np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None

    def __post_init__(self):
        # If id is not set, generate a random one
        if self.id is None:
            self.id = compute_mdhash_id(str(time.time_ns()), prefix="emb")


@dataclass(slots=True)
class EmbeddingHit:
    """
    Vector query hit.

    :param id: Matched record identifier.
    :param distance: Similarity/distance score to query embedding.
    :param metadata: Additional payload.
    """
    id: str
    distance: float
    metadata: Dict[str, Any] = field(default_factory=dict)
