import asyncio
from pathlib import Path

import pytest

from ragu.utils.ragu_utils import (
    attach_async_contexts,
    always_get_an_event_loop,
    compute_mdhash_id,
    deprecated,
    get_disk_cache,
    read_text_from_files,
    save_args_on_exception,
)


def test_compute_mdhash_id_is_deterministic():
    first = compute_mdhash_id("a", "b", x="1", y="2")
    second = compute_mdhash_id("a", "b", y="2", x="1")
    assert first == second


def test_compute_mdhash_id_prefix():
    value = compute_mdhash_id("data", prefix="ent-")
    assert value.startswith("ent-")
    assert len(value) == len("ent-") + 32


def test_always_get_an_event_loop_returns_open_loop():
    loop = always_get_an_event_loop()
    assert isinstance(loop, asyncio.AbstractEventLoop)
    assert not loop.is_closed()


def test_get_disk_cache_returns_shared_instance(tmp_path):
    cache_dir = tmp_path / "disk_cache"
    first = get_disk_cache(cache_dir)
    first["key"] = "value"
    second = get_disk_cache(cache_dir)
    assert second["key"] == "value"


@pytest.mark.asyncio
async def test_attach_async_contexts_wraps_function_execution():
    events: list[str] = []

    class _Ctx:
        async def __aenter__(self):
            events.append("enter")
            return self

        async def __aexit__(self, exc_type, exc, tb):
            events.append("exit")
            return False

    async def fn(value: int) -> int:
        events.append("run")
        return value * 2

    wrapped = attach_async_contexts(fn, _Ctx())
    result = await wrapped(5)

    assert result == 10
    assert events == ["enter", "run", "exit"]


@pytest.mark.asyncio
async def test_save_args_on_exception_stores_call_data():
    storage: dict[str, object] = {}

    async def fn(x: int, *, y: int) -> int:
        raise ValueError("boom")

    wrapped = save_args_on_exception(fn, storage)

    with pytest.raises(ValueError):
        await wrapped(1, y=2)

    assert len(storage) == 1
    payload = next(iter(storage.values()))
    assert payload["function_name"].endswith("fn")
    assert payload["args"] == (1,)
    assert payload["kwargs"] == {"y": 2}


def test_read_text_from_files_reads_nested_files(tmp_path):
    root = Path(tmp_path) / "docs"
    nested = root / "sub"
    nested.mkdir(parents=True)
    (root / "a.txt").write_text("alpha", encoding="utf-8")
    (nested / "b.md").write_text("beta", encoding="utf-8")

    values = read_text_from_files(root, file_extensions={".txt", ".md"})
    assert sorted(values) == ["alpha", "beta"]


class TestDeprecated:
    def test_function_warns_and_still_works(self):
        @deprecated(replacement="new_one")
        def old(x):
            return x * 2

        with pytest.warns(DeprecationWarning, match="old is deprecated. Use new_one instead."):
            assert old(3) == 6

    async def test_coroutine_warns_and_still_works(self):
        @deprecated()
        async def old():
            return "done"

        with pytest.warns(DeprecationWarning, match="old is deprecated."):
            assert await old() == "done"

    def test_class_stays_a_class(self):
        """Wrapping a class in a factory function would break isinstance and subclassing."""
        @deprecated(replacement="NewOne")
        class Old:
            """Kept docstring."""

            def __init__(self, x=0):
                self.x = x

            def hello(self):
                return "hi"

        assert isinstance(Old, type)
        assert Old.__name__ == "Old"
        assert Old.__doc__ == "Kept docstring."

        with pytest.warns(DeprecationWarning, match="Old is deprecated. Use NewOne instead."):
            instance = Old(2)

        assert isinstance(instance, Old)
        assert instance.x == 2
        assert instance.hello() == "hi"

    def test_subclassing_a_deprecated_class_works_and_warns(self):
        @deprecated()
        class Old:
            def __init__(self, x):
                self.x = x

        class Child(Old):
            def __init__(self):
                super().__init__(5)

        with pytest.warns(DeprecationWarning):
            child = Child()

        assert child.x == 5
        assert isinstance(child, Old)

    def test_class_without_replacement(self):
        @deprecated()
        class Old:
            pass

        with pytest.warns(DeprecationWarning, match=r"Old is deprecated\.$"):
            Old()
