import os
from datetime import datetime
from pathlib import Path
from typing import Any

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
    __instance = None
    _current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    _working_dir = os.path.join(os.path.join(os.getcwd(), "ragu_working_dir"), _current_time)

    language: str = "english"

    tokenizer_embedder_backend: str = "tiktoken"
    tokenizer_llm_backend: str = "tiktoken"
    tokenizer_embedder_name: str = "text-embedding-3-large"
    tokenizer_llm_name: str = "gpt-4o"
    embedder_token_limit: int = 8_192
    llm_token_limit: int = 32_798
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


Settings = GlobalSettings()




