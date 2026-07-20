# Based on https://github.com/gusye1234/nano-graphrag/blob/main/nano_graphrag/_storage/vdb_nanovectordb.py

import json
import os
import tempfile
from typing import List, TypeVar, Union
from typing_extensions import override

from ragu.common.global_parameters import Settings
from ragu.storage.base_storage import BaseKVStorage


T = TypeVar("T")

class JsonKVStorage(BaseKVStorage[dict[str, T]]):
    """
    Key-value storage implementation using a local JSON file.

    This class provides a simple persistent storage backend.
    All data is loaded into memory at initialization and written back to disk upon updates.
    """

    def __init__(self, storage_folder: str = Settings.storage_folder, filename: str = "kv_store.json"):
        """
        Initialize the JSON key-value storage.

        Creates a new JSON file if one does not exist, otherwise loads existing data.

        :param storage_folder: Path to the folder where the storage file will be located.
        :param filename: Name of the JSON file used for storage.
        """
        self.filename = os.path.join(storage_folder, filename)
        if not os.path.exists(self.filename):
            self.data: dict[str, dict[str, T]] = {}
            with open(self.filename, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        else:
            with open(self.filename, encoding="utf-8") as f:
                self.data = json.load(f)

    @override
    async def all_keys(self) -> list[str]:
        """
        Return a list of all keys currently stored in the JSON file.

        :return: List of keys present in the store.
        """
        return list(self.data.keys())

    @override
    async def get_by_id(self, id: str) -> Union[dict[str, T], None]:
        """
        Retrieve a record by its unique identifier.

        :param id: Unique identifier key.
        :return: The stored value or ``None`` if not found.
        """
        return self.data.get(id, None)

    @override
    async def get_by_ids(self, ids: list[str], fields: Union[set[str], None] = None) -> List[Union[dict[str, T], None]]:
        """
        Retrieve multiple records by their identifiers.

        Optionally return only specified fields from each record.

        :param ids: Iterable of record IDs to fetch.
        :param fields: Optional list of fields to include in the result.
        :return: List of records or field-filtered dictionaries.
        """
        if fields is None:
            return [self.data.get(id, None) for id in ids]
        return [
            (
                {k: v for k, v in self.data[id].items() if k in fields}
                if self.data.get(id, None)
                else None
            )
            for id in ids
        ]

    @override
    async def filter_keys(self, data: list[str]) -> set[str]:
        """
        Return a subset of keys that are not yet present in the store.

        :param data: List of keys to check.
        :return: Set of keys missing from the storage.
        """
        return set([s for s in data if s not in self.data])

    @override
    async def upsert(self, data: dict[str, dict[str, T]]):
        """
        Insert or update one or more key-value pairs in the store.

        :param data: Dictionary of key-value mappings to update.
        """
        self.data.update(data)

    @override
    async def delete(self, ids: list[str]) -> None:
        """
        Delete multiple records by their IDs from the key-value store.

        :param ids: List of IDs to remove.
        :type ids: list[str]
        """
        for id_ in ids:
            self.data.pop(id_, None)

    @override
    async def drop(self):
        """
        Remove all records from the store (in-memory only).
        """
        self.data = {}

    async def index_start_callback(self):
        """
        Pre-index hook for interface compatibility.
        """
        pass

    async def query_done_callback(self):
        """
        Post-query hook for interface compatibility.
        """
        pass

    async def index_done_callback(self):
        """
        Persist the current in-memory data to disk.
        """
        directory = os.path.dirname(self.filename) or "."
        os.makedirs(directory, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=directory,
            prefix=os.path.basename(self.filename) + ".",
            suffix=".tmp",
            delete=False,
        )
        try:
            with handle as file:
                json.dump(self.data, file, indent=2, ensure_ascii=False)
                file.flush()
                os.fsync(file.fileno())
            os.replace(handle.name, self.filename)
        except BaseException:
            if os.path.exists(handle.name):
                os.unlink(handle.name)
            raise
