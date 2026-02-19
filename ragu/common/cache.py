import asyncio
from collections.abc import MutableMapping
import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel
from diskcache import Index # pyright: ignore[reportMissingTypeStubs]

from ragu.common.global_parameters import DEFAULT_FILENAMES, Settings
from ragu.common.logger import logger
from ragu.common.prompts import ChatMessages
from ragu.utils.ragu_utils import compute_mdhash_id


@dataclass(frozen=True, slots=True)
class PendingRequest:
    """
    Represents a request pending generation (not found in cache).
    """
    index: int
    messages: ChatMessages
    cache_key: str

_schema_cache: dict[type[BaseModel], str] = {}

def make_llm_cache_key(
    content: str,
    model_name: Optional[str] = None,
    schema: Optional[Type[BaseModel]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build a deterministic cache key from LLM request parameters.

    :param model_name: Model name used for generation.
    :param schema: Optional Pydantic schema class.
    :param kwargs: Additional API parameters.
    :return: A unique cache key string.
    """
    key_parts = [content]

    if model_name:
        key_parts.append(f"[model]: {model_name}")

    if schema is not None:
        if not (schema_str := _schema_cache.get(schema, None)):
            print(f'Cache miss for schema {schema}')
            schema_str = json.dumps(schema.model_json_schema(), sort_keys=True)
            _schema_cache[schema] = schema_str
        key_parts.append(schema_str)

    if kwargs:
        sorted_kwargs = sorted(kwargs.items())
        kwargs_str = json.dumps(sorted_kwargs, sort_keys=True)
        key_parts.append(f"[kwargs]: {kwargs_str}")

    return compute_mdhash_id(*key_parts, prefix="llm-cache-")

def make_embedding_cache_key(text: str, model_name: str) -> str:
    """
    Build a cache key from text and model name.
    """
    key_str = f"[model]: {model_name}\n[text]: {text}"
    return compute_mdhash_id(key_str, prefix="emb-cache-")

class TextCache:
    """
    Key-value cache for LLM responses with automatic BaseModel serialization.

    Supports both string and Pydantic BaseModel responses. When a schema is provided,
    automatically serializes/deserializes BaseModel instances.
    """

    __instance = None

    def __init__(
        self,
        cache_path: str | Path | None = None,
        *,
        flush_every_n_writes: int = 10,
    ) -> None:
        """
        Initialize async LLM cache.

        :param cache_path: Path to cache file. Defaults to storage_folder/llm_cache.json
        :param flush_every_n_writes: Write to disk after N cache updates.
        """
        self.flush_every_n_writes = max(1, flush_every_n_writes)
        self._mem_cache: Dict[str, Any] = {}
        self._pending_disk_writes = 0
        self._write_lock = asyncio.Lock()

        if cache_path is None:
            Settings.init_storage_folder()
            default_name = DEFAULT_FILENAMES["llm_cache_file_name"]
            cache_path = Path(Settings.storage_folder) / default_name

        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path = cache_path
        self._load_cache()

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super(TextCache, cls).__new__(cls)
            return cls.__instance
        else:
            return cls.__instance

    def _load_cache(self) -> None:
        """
        Load cache from disk into memory.
        """
        if not self._cache_path.exists():
            return

        try:
            with self._cache_path.open("r", encoding="utf-8") as f:
                cache = json.load(f)
                if isinstance(cache, dict):
                    self._mem_cache = cache
        except Exception as e:
            logger.warning(f"Failed to load cache from {self._cache_path}: {e}")

    async def flush_cache(self) -> None:
        """
        Flush cache to disk.
        """
        if not self._mem_cache:
            self._pending_disk_writes = 0
            return

        async with self._write_lock:
            try:
                tmp_path = self._cache_path.with_suffix(self._cache_path.suffix + ".tmp")

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    self._write_cache_file,
                    tmp_path,
                )

                tmp_path.replace(self._cache_path)
                self._pending_disk_writes = 0
            except Exception as e:
                logger.warning(f"Failed to flush cache to {self._cache_path}: {e}")

    def _write_cache_file(self, path: Path) -> None:
        """
        Write cache to file.
        """
        with path.open("w", encoding="utf-8") as f:
            json.dump(self._mem_cache, f, ensure_ascii=False, indent=2)

    async def get(
        self,
        key: str,
        *,
        schema: Optional[Type[BaseModel]] = None,
    ) -> Optional[Any]:
        """
        Retrieve a value from cache.

        :param key: Cache key.
        :param schema: Optional Pydantic schema to reconstruct BaseModel.
        :return: Cached value (string or BaseModel), or None if not found.
        """
        cached = self._mem_cache.get(key)
        if cached is None:
            return None

        # Extract response from payload if it's a structured cache entry
        if isinstance(cached, dict) and "response" in cached:
            response = cached["response"]
        else:
            # Backward compatibility: old cache entries without payload structure
            response = cached

        # If schema provided and response is a dict, reconstruct BaseModel
        if schema is not None and isinstance(response, dict):
            try:
                return schema.model_validate(response)
            except Exception as e:
                logger.warning(f"Failed to reconstruct {schema.__name__} from cache: {e}")
                return None

        return response

    async def set(
        self,
        key: str,
        value: Any,
        **additional_payload,
    ) -> None:
        """
        Store a value in cache.

        :param key: Cache key.
        :param value: Value to cache (string or BaseModel).
        """
        # Serialize BaseModel to dict for storage
        if isinstance(value, BaseModel):
            cached_value = value.model_dump()
        else:
            cached_value = value

        payload: Dict[str, Any] = {
            "response": cached_value,
            "additional_payload": additional_payload,
            "time": datetime.now(timezone.utc).isoformat()
        }

        self._mem_cache[key] = payload
        self._pending_disk_writes += 1

        if self._pending_disk_writes >= self.flush_every_n_writes:
            await self.flush_cache()

    async def close(self) -> None:
        """
        Flush any pending writes and close cache.
        """
        if self._pending_disk_writes > 0:
            await self.flush_cache()


class EmbeddingCache:
    """
    Async cache specifically for embeddings (lists of floats).

    Uses pickle serialization for efficient storage of numeric data.
    Separate from LLM cache to allow independent management and different storage strategies.
    """

    def __init__(
        self,
        cache_path: str | Path | None = None,
        *,
        flush_every_n_writes: int = 50,
    ) -> None:
        """
        Initialize async embedding cache.

        :param cache_path: Path to cache file. Defaults to storage_folder/embedding_cache.pkl
        :param flush_every_n_writes: Write to disk after N cache updates.
        """
        self.flush_every_n_writes = max(1, flush_every_n_writes)

        self._mem_cache: Dict[str, List[float]] = {}
        self._pending_disk_writes = 0
        self._write_lock = asyncio.Lock()

        if cache_path is None:
            Settings.init_storage_folder()
            default_name = DEFAULT_FILENAMES["embedding_cache_file_name"]
            cache_path = Path(Settings.storage_folder) / default_name

        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path = cache_path
        self._load_cache()

    def _load_cache(self) -> None:
        """
        Load cache from disk into memory.
        """
        if not self._cache_path.exists():
            return

        try:
            with self._cache_path.open("rb") as f:
                cache = pickle.load(f)
                if isinstance(cache, dict):
                    self._mem_cache = cache
        except Exception as e:
            logger.warning(f"Failed to load embedding cache from {self._cache_path}: {e}")

    async def flush_cache(self) -> None:
        """
        Flush cache to disk.
        """
        if not self._mem_cache:
            self._pending_disk_writes = 0
            return

        async with self._write_lock:
            try:
                tmp_path = self._cache_path.with_suffix(self._cache_path.suffix + ".tmp")

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    self._write_cache_file,
                    tmp_path,
                )

                tmp_path.replace(self._cache_path)
                self._pending_disk_writes = 0
            except Exception as e:
                logger.warning(f"Failed to flush embedding cache to {self._cache_path}: {e}")

    def _write_cache_file(self, path: Path) -> None:
        """
        Write cache to file.
        """
        with path.open("wb") as f:
            pickle.dump(self._mem_cache, f, protocol=pickle.HIGHEST_PROTOCOL) # type: ignore

    async def get(self, key: str) -> Optional[List[float]]:
        """
        Retrieve an embedding from cache.

        :param key: Cache key.
        :return: Cached embedding (list of floats), or None if not found.
        """
        return self._mem_cache.get(key)

    async def set(self, key: str, embedding: List[float]) -> None:
        """
        Store an embedding in cache.

        :param key: Cache key.
        :param embedding: Embedding vector to cache (list of floats).
        """
        self._mem_cache[key] = embedding
        self._pending_disk_writes += 1

        # Flush to disk if threshold reached
        if self._pending_disk_writes >= self.flush_every_n_writes:
            await self.flush_cache()

    async def close(self) -> None:
        """
        Flush any pending writes and close cache.
        """
        if self._pending_disk_writes > 0:
            await self.flush_cache()


_CACHES: dict[str, Index] = {}

def get_cache(dir: str | Path) -> MutableMapping[str, Any]:
    """Get or create a key-value cache which uses the specified directory."""
    dir = str(dir)
    if (cache := _CACHES.get(dir, None)) is None:
        cache = _CACHES[dir] = Index(dir)
    return cache