"""Concurrency contracts for the adapter-independent element scheduler."""

import asyncio
import random
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from ragu.graph._locks import _ElementLocks, _conflicts, _covers


async def _hold_until(
    locks,
    requirements,
    entered,
    release,
    attempting=None,
):
    if attempting is not None:
        attempting.set()
    async with locks.hold(requirements):
        entered.set()
        await release.wait()


async def _wait(event):
    await asyncio.wait_for(event.wait(), timeout=2)


def test_wildcard_lookup_matches_footprint_intersection_and_coverage():
    """Compare indexed lookup with the direct set-intersection definition."""
    rng = random.Random(0)
    keys = [(namespace, item_id) for namespace in ("node", "edge", "*")
            for item_id in ("a", "b", "*")]
    for _ in range(300):
        left = {key: rng.choice((False, True)) for key in rng.sample(keys, rng.randrange(10))}
        right = {key: rng.choice((False, True)) for key in rng.sample(keys, rng.randrange(10))}
        conflict = any(
            (left_write or right_write)
            and all(a == b or a == "*" or b == "*" for a, b in zip(left_key, right_key))
            for left_key, left_write in left.items()
            for right_key, right_write in right.items()
        )
        covered = all(
            any(
                (held_write or not requested_write)
                and all(a == "*" or a == b for a, b in zip(held_key, requested_key))
                for held_key, held_write in left.items()
            )
            for requested_key, requested_write in right.items()
        )
        assert _conflicts(left, right) == conflict
        assert _conflicts(right, left) == conflict
        assert _covers(left, right) == covered


async def test_readers_share_an_element_and_disjoint_writes_bypass_waiters():
    locks = _ElementLocks()
    reader_entered, writer_entered = asyncio.Event(), asyncio.Event()
    writer_attempting, release = asyncio.Event(), asyncio.Event()

    async with locks.hold({("node", "a"): False}):
        reader = asyncio.create_task(_hold_until(
            locks, {("node", "a"): False}, reader_entered, release,
        ))
        await _wait(reader_entered)
        writer = asyncio.create_task(_hold_until(
            locks, {("node", "a"): True}, writer_entered, release, writer_attempting,
        ))
        await _wait(writer_attempting)

        disjoint_entered = asyncio.Event()
        disjoint = asyncio.create_task(_hold_until(
            locks, {("node", "c"): True}, disjoint_entered, release,
        ))
        await _wait(disjoint_entered)
        assert not writer_entered.is_set()
        release.set()
        await reader
        await disjoint

    await asyncio.wait_for(writer, timeout=2)
    assert writer_entered.is_set()
    assert locks._active == {}
    assert locks._waiting == []


async def test_waiting_writer_precedes_later_conflicting_reader():
    locks = _ElementLocks()
    writer_attempting, reader_attempting = asyncio.Event(), asyncio.Event()
    writer_entered, reader_entered = asyncio.Event(), asyncio.Event()
    writer_release, reader_release = asyncio.Event(), asyncio.Event()

    async with locks.hold({("node", "a"): False}):
        writer = asyncio.create_task(_hold_until(
            locks, {("node", "a"): True}, writer_entered, writer_release, writer_attempting,
        ))
        await _wait(writer_attempting)
        reader = asyncio.create_task(_hold_until(
            locks, {("node", "a"): False}, reader_entered, reader_release, reader_attempting,
        ))
        await _wait(reader_attempting)
        assert not writer_entered.is_set()
        assert not reader_entered.is_set()

    await _wait(writer_entered)
    assert not reader_entered.is_set()
    writer_release.set()
    await _wait(reader_entered)
    reader_release.set()
    await asyncio.gather(writer, reader)


async def test_multi_element_acquisition_is_atomic_regardless_of_key_order():
    locks = _ElementLocks()
    attempted, entered, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async with locks.hold({("node", "a"): True, ("node", "b"): True}):
        reverse = asyncio.create_task(_hold_until(
            locks,
            {("node", "b"): True, ("node", "a"): True},
            entered,
            release,
            attempted,
        ))
        await _wait(attempted)
        assert not entered.is_set()

    await _wait(entered)
    release.set()
    await reverse


@pytest.mark.parametrize("wildcard", [("*", "*"), ("node", "*"), ("*", "a")])
async def test_wildcards_conflict_with_matching_elements(wildcard):
    locks = _ElementLocks()
    attempted, entered, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async with locks.hold({wildcard: True}):
        waiter = asyncio.create_task(_hold_until(
            locks, {("node", "a"): False}, entered, release, attempted,
        ))
        await _wait(attempted)
        assert not entered.is_set()
    await _wait(entered)
    release.set()
    await waiter


async def test_namespace_wildcards_leave_other_namespaces_available():
    locks = _ElementLocks()
    entered, release = asyncio.Event(), asyncio.Event()
    async with locks.hold({("node", "*"): True}):
        unrelated = asyncio.create_task(_hold_until(
            locks, {("edge", "a"): True}, entered, release,
        ))
        await _wait(entered)
        release.set()
        await unrelated


async def test_nested_scopes_require_existing_coverage_and_preserve_outer_grants():
    locks = _ElementLocks()
    assert locks.held_requirements() is None
    async with locks.hold({("node", "*"): True, ("chunk", "x"): False}):
        assert locks.covers({("node", "a"): False})
        assert locks.covers({("node", "a"): True})
        async with locks.hold({("node", "a"): False}):
            assert locks.covers({("node", "*"): True})
        assert locks.covers({("node", "b"): True})
        snapshot = locks.held_requirements()
        snapshot.clear()
        assert locks.covers({("node", "b"): True})

        with pytest.raises(ValueError, match="Nested lock scopes"):
            async with locks.hold({("chunk", "x"): True}):
                pytest.fail("Read grants cannot be upgraded")
        with pytest.raises(ValueError, match="Nested lock scopes"):
            async with locks.hold({("edge", "a"): False}):
                pytest.fail("Nested scopes cannot introduce new keys")
    assert locks.held_requirements() is None


async def test_child_task_does_not_inherit_parent_ownership():
    locks = _ElementLocks()
    attempted, entered, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async with locks.hold({("node", "a"): True}):
        child = asyncio.create_task(_hold_until(
            locks, {("node", "a"): True}, entered, release, attempted,
        ))
        await _wait(attempted)
        assert not entered.is_set()
    await _wait(entered)
    release.set()
    await child


async def test_cancelling_waiter_unblocks_later_readers():
    locks = _ElementLocks()
    writer_attempting, writer_entered = asyncio.Event(), asyncio.Event()
    reader_attempting, reader_entered = asyncio.Event(), asyncio.Event()
    release = asyncio.Event()
    async with locks.hold({("node", "a"): False}):
        writer = asyncio.create_task(_hold_until(
            locks, {("node", "a"): True}, writer_entered, release, writer_attempting,
        ))
        await _wait(writer_attempting)
        reader = asyncio.create_task(_hold_until(
            locks, {("node", "a"): False}, reader_entered, release, reader_attempting,
        ))
        await _wait(reader_attempting)
        assert not reader_entered.is_set()
        writer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await writer
        await _wait(reader_entered)
        release.set()
        await reader
    assert not writer_entered.is_set()
    assert locks._waiting == []


async def test_cancellation_after_grant_before_wakeup_releases_grants():
    locks = _ElementLocks()
    attempted, entered, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async with locks.hold({("node", "a"): True}):
        waiter = asyncio.create_task(_hold_until(
            locks, {("node", "a"): True}, entered, release, attempted,
        ))
        await _wait(attempted)

    # Exiting the holder grants the waiter and schedules its future's wake-up.
    # Cancel before that callback runs, exercising the grant/cancel race.
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert not entered.is_set()
    assert locks._active == {}
    assert locks._waiting == []
    async with locks.hold({("node", "a"): True}):
        pass


async def test_exception_and_owner_cancellation_release_grants():
    locks = _ElementLocks()
    with pytest.raises(RuntimeError, match="operation failed"):
        async with locks.hold({("node", "a"): True}):
            raise RuntimeError("operation failed")

    entered, release = asyncio.Event(), asyncio.Event()
    owner = asyncio.create_task(_hold_until(
        locks, {("node", "a"): True}, entered, release,
    ))
    await _wait(entered)
    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner
    async with locks.hold({("node", "a"): True}):
        pass
    assert locks._active == {}
    assert locks._waiting == []


def test_locks_coordinate_threads_with_separate_event_loops():
    locks = _ElementLocks()
    first_entered, first_release = threading.Event(), threading.Event()
    second_attempting, second_entered, disjoint_entered = (
        threading.Event(), threading.Event(), threading.Event()
    )

    async def first_loop():
        async with locks.hold({("node", "a"): False}):
            first_entered.set()
            assert await asyncio.to_thread(first_release.wait, 3)

    async def second_loop():
        async def conflicting_writer():
            second_attempting.set()
            async with locks.hold({("node", "a"): True}):
                second_entered.set()

        async def unrelated_writer():
            async with locks.hold({("node", "c"): True}):
                disjoint_entered.set()

        await asyncio.gather(conflicting_writer(), unrelated_writer())

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(asyncio.run, first_loop())
        try:
            assert first_entered.wait(3)
            second = executor.submit(asyncio.run, second_loop())
            assert second_attempting.wait(3)
            assert disjoint_entered.wait(3)
            assert not second_entered.is_set()
        finally:
            first_release.set()
        first.result(timeout=3)
        second.result(timeout=3)
    assert second_entered.is_set()
    assert locks.held_requirements() is None
    assert locks._active == {}
    assert locks._waiting == []
