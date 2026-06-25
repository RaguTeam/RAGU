import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints

from loguru import logger


DEFAULT_FILENAMES = {
    "community_kv_storage_name": "kv_community.json",
    "chunks_kv_storage_name": "kv_chunks.json",
    "entity_vdb_name": "vdb_entity.json",
    "relation_vdb_name": "vdb_relation.json",
    "chunk_vdb_name": "vdb_chunk.json",
    "knowledge_graph_storage_name": "knowledge_graph.gml",
    "community_summary_kv_storage_name": "kv_community_summary.json",
    "llm_cache_file_name": "llm_cache.jsonl",
    "embedding_cache_file_name": "embedding_cache.pkl",
}


class GlobalSettings:
    """Process-wide singleton holding RAGU runtime defaults.

    The user-configurable fields carry type annotations and are the only
    attributes participating in :meth:`save` / :meth:`load`. Runtime state
    such as ``_working_dir`` (the storage folder, which embeds a timestamp),
    ``cache_path`` / ``debug_errors_path`` (local, machine-specific paths),
    and the singleton handle are intentionally excluded from serialization.
    """

    __instance = None
    _current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    _working_dir = os.path.join(os.path.join(os.getcwd(), "ragu_working_dir"), _current_time)
    _cache_path: str | None = None
    _debug_errors_path: str | None = None

    language: str = "english"

    tokenizer_embedder_backend: Literal["tiktoken", "local"] = "tiktoken"
    tokenizer_llm_backend: Literal["tiktoken", "local"] = "tiktoken"
    tokenizer_embedder_name: str = "text-embedding-3-large"
    tokenizer_llm_name: str = "gpt-4o"
    embedder_token_limit: int = 8_192
    llm_context_token_limit: int = 30_000

    def __new__(cls, *args: Any, **kwargs: Any):
        if cls.__instance is None:
            cls.__instance = super(GlobalSettings, cls).__new__(cls)
            return cls.__instance
        else:
            return cls.__instance

    @property
    def storage_folder(self) -> str:
        return self._working_dir

    @storage_folder.setter
    def storage_folder(self, path: str | Path):
        self._working_dir = str(path)

    def init_storage_folder(self):
        if not os.path.exists(self._working_dir):
            logger.info(f"Creating folder for current run: {self._working_dir}")
            os.makedirs(self._working_dir, exist_ok=True)

    @property
    def cache_path(self) -> str | None:
        """Stable directory for the LLM/embedding response cache, or ``None``.

        Not serialized: it is a local, machine-specific path. Defaults to
        ``None`` (caching disabled). Deliberately independent of
        :attr:`storage_folder`, which is per-run and timestamped.
        """
        return self._cache_path

    @cache_path.setter
    def cache_path(self, value: str | Path | None) -> None:
        self._cache_path = str(value) if value is not None else None

    @property
    def debug_errors_path(self) -> str | None:
        """Stable directory for the error-debug store, or ``None``.

        Not serialized: local, machine-specific path. Defaults to ``None``
        (error-args capture disabled).
        """
        return self._debug_errors_path

    @debug_errors_path.setter
    def debug_errors_path(self, value: str | Path | None) -> None:
        self._debug_errors_path = str(value) if value is not None else None

    def save(self, path: str | Path) -> None:
        """Serialize the user-configurable settings to a JSON file.

        Only annotated public class attributes are written. ``storage_folder``,
        ``cache_path`` and ``debug_errors_path`` are deliberately excluded: by
        default ``storage_folder`` embeds a runtime timestamp, and the cache /
        debug paths are local and machine-specific, so restoring any of them
        silently would redirect subsequent writes into a stale or nonexistent
        location. Manage these paths explicitly in application code if they
        must be reproduced.

        :param path: Path to the output JSON file. Parent directories are
            created automatically.
        """
        data = self._serializable_dict()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Saved global settings to {output_path}")

    def load(self, path: str | Path) -> None:
        """Load settings from a JSON file produced by :meth:`save`.

        Each value is validated against the declared type hints (including
        the ``Literal`` backend choices); a mismatch raises ``ValueError``.
        Unknown keys are reported via ``logger.warning`` and ignored so that
        files written by a newer RAGU version remain loadable.

        Mutates the singleton in place. ``storage_folder``, ``cache_path`` and
        ``debug_errors_path`` are never touched.

        :param path: Path to the JSON file to read.
        :raises ValueError: if the file cannot be read, is not a JSON
            object, or a value fails validation.
        """
        input_path = Path(path)
        try:
            raw = input_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(
                f"Failed to read settings file '{input_path}': {e}"
            ) from e
        if not isinstance(data, dict):
            raise ValueError(
                f"Settings file '{input_path}' must contain a JSON object, "
                f"got {type(data).__name__}."
            )
        self._apply_dict(data, source=str(input_path))
        logger.info(f"Loaded global settings from {input_path}")

    def _serializable_dict(self) -> dict[str, Any]:
        """Return the current values of all serializable fields.

        Reads through the singleton instance (``self``) so that user
        overrides such as ``Settings.embedder_token_limit = 512`` — which
        create instance attributes shadowing the class defaults — are
        captured rather than the original class-level defaults.
        """
        hints = get_type_hints(type(self))
        return {
            name: getattr(self, name)
            for name in hints
            if not name.startswith("_")
        }

    def _apply_dict(
        self,
        data: dict[str, Any],
        source: str,
    ) -> None:
        """Validate and apply a settings mapping onto the singleton."""
        hints = get_type_hints(type(self))
        serializable = {
            name for name in hints if not name.startswith("_")
        }
        for key, value in data.items():
            if key not in serializable:
                logger.warning(
                    f"Unknown setting '{key}' in {source}; ignored."
                )
                continue
            self._validate_value(key, value, hints[key], source)
            setattr(self, key, value)

    @staticmethod
    def _validate_value(
        name: str,
        value: Any,
        annotation: Any,
        source: str,
    ) -> None:
        """Validate a single value against its declared annotation.

        :raises ValueError: if the value does not match the annotation.
        """
        if get_origin(annotation) is Literal:
            allowed = get_args(annotation)
            if value not in allowed:
                raise ValueError(
                    f"Invalid value for '{name}' in {source}: "
                    f"expected one of {list(allowed)}, got {value!r}."
                )
            return
        if annotation is int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"Invalid value for '{name}' in {source}: "
                    f"expected int, got {value!r}."
                )
            if value <= 0:
                raise ValueError(
                    f"Invalid value for '{name}' in {source}: "
                    f"expected a positive integer, got {value}."
                )
            return
        if annotation is str:
            if not isinstance(value, str):
                raise ValueError(
                    f"Invalid value for '{name}' in {source}: "
                    f"expected str, got {value!r}."
                )
            return
        logger.debug(
            f"No validation rule for '{name}' "
            f"(annotation {annotation!r}); accepted as-is."
        )


Settings = GlobalSettings()
