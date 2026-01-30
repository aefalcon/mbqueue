import sys
import threading
import time

import pytest

import mbqueue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HAS_SHUTDOWN = sys.version_info >= (3, 13)


def _skip_unless_shutdown():
    if not HAS_SHUTDOWN:
        pytest.skip("shutdown requires Python 3.13+")


# ---------------------------------------------------------------------------
# Basic FIFO
# ---------------------------------------------------------------------------


class TestBasicFIFO:
    def test_put_get_single(self):
        q = mbqueue.Queue()
        q.put("hello")
        assert q.get() == "hello"

    def test_fifo_order(self):
        q = mbqueue.Queue()
        for i in range(5):
            q.put(i)
        result = [q.get() for _ in range(5)]
        assert result == [0, 1, 2, 3, 4]

    def test_various_types(self):
        q = mbqueue.Queue()
        items = [42, 3.14, "str", None, [1, 2], {"a": 1}]
        for item in items:
            q.put(item)
        for item in items:
            assert q.get() == item


# ---------------------------------------------------------------------------
# Size / capacity
# ---------------------------------------------------------------------------


class TestSizeCapacity:
    def test_qsize(self):
        q = mbqueue.Queue()
        assert q.qsize() == 0
        q.put(1)
        assert q.qsize() == 1
        q.put(2)
        assert q.qsize() == 2
        q.get()
        assert q.qsize() == 1

    def test_empty(self):
        q = mbqueue.Queue()
        assert q.empty()
        q.put(1)
        assert not q.empty()

    def test_full_unbounded(self):
        q = mbqueue.Queue()
        assert not q.full()
        for i in range(100):
            q.put(i)
        assert not q.full()

    def test_full_bounded(self):
        q = mbqueue.Queue(maxsize=2)
        assert not q.full()
        q.put(1)
        assert not q.full()
        q.put(2)
        assert q.full()

    def test_maxsize_property(self):
        q = mbqueue.Queue(maxsize=10)
        assert q.maxsize == 10

    def test_maxsize_zero(self):
        q = mbqueue.Queue(maxsize=0)
        assert q.maxsize == 0
        assert not q.full()

    def test_maxsize_negative(self):
        q = mbqueue.Queue(maxsize=-1)
        assert q.maxsize == -1
        assert not q.full()


# ---------------------------------------------------------------------------
# Exceptions: Empty / Full
# ---------------------------------------------------------------------------


class TestExceptions:
    def test_get_nowait_empty(self):
        q = mbqueue.Queue()
        with pytest.raises(mbqueue.Empty):
            q.get_nowait()

    def test_get_nonblocking_empty(self):
        q = mbqueue.Queue()
        with pytest.raises(mbqueue.Empty):
            q.get(block=False)

    def test_put_nowait_full(self):
        q = mbqueue.Queue(maxsize=1)
        q.put(1)
        with pytest.raises(mbqueue.Full):
            q.put_nowait(2)

    def test_put_nonblocking_full(self):
        q = mbqueue.Queue(maxsize=1)
        q.put(1)
        with pytest.raises(mbqueue.Full):
            q.put(2, block=False)


# ---------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------


class TestTimeouts:
    def test_get_timeout(self):
        q = mbqueue.Queue()
        start = time.monotonic()
        with pytest.raises(mbqueue.Empty):
            q.get(timeout=0.05)
        elapsed = time.monotonic() - start
        assert elapsed >= 0.04

    def test_put_timeout(self):
        q = mbqueue.Queue(maxsize=1)
        q.put(1)
        start = time.monotonic()
        with pytest.raises(mbqueue.Full):
            q.put(2, timeout=0.05)
        elapsed = time.monotonic() - start
        assert elapsed >= 0.04

    def test_get_timeout_none_is_blocking(self):
        """timeout=None should behave like blocking (tested by unblocking from thread)."""
        q = mbqueue.Queue()

        def feeder():
            time.sleep(0.05)
            q.put("data")

        t = threading.Thread(target=feeder)
        t.start()
        result = q.get(timeout=None)
        assert result == "data"
        t.join()

    def test_negative_timeout_is_infinite(self):
        """A negative timeout should behave as no timeout (infinite wait)."""
        q = mbqueue.Queue()

        def feeder():
            time.sleep(0.05)
            q.put("value")

        t = threading.Thread(target=feeder)
        t.start()
        result = q.get(timeout=-1)
        assert result == "value"
        t.join()


# ---------------------------------------------------------------------------
# task_done / join
# ---------------------------------------------------------------------------


class TestTaskDoneJoin:
    def test_task_done_too_many(self):
        q = mbqueue.Queue()
        with pytest.raises(ValueError):
            q.task_done()

    def test_task_done_too_many_after_done(self):
        q = mbqueue.Queue()
        q.put(1)
        q.get()
        q.task_done()
        with pytest.raises(ValueError):
            q.task_done()

    def test_join_empty_queue(self):
        """join() on a fresh queue should return immediately."""
        q = mbqueue.Queue()
        q.join()  # should not block

    def test_join_unblocks(self):
        q = mbqueue.Queue()
        q.put(1)
        q.put(2)

        results = []

        def worker():
            while True:
                item = q.get()
                results.append(item)
                q.task_done()
                if item == 2:
                    break

        t = threading.Thread(target=worker)
        t.start()
        q.join()
        t.join()
        assert sorted(results) == [1, 2]


# ---------------------------------------------------------------------------
# Blocking producer/consumer
# ---------------------------------------------------------------------------


class TestBlocking:
    def test_put_blocks_on_full(self):
        q = mbqueue.Queue(maxsize=1)
        q.put("a")
        done = threading.Event()

        def consumer():
            time.sleep(0.05)
            q.get()
            q.task_done()

        t = threading.Thread(target=consumer)
        t.start()
        q.put("b")  # should block until consumer frees space
        t.join()
        assert q.qsize() == 1

    def test_get_blocks_on_empty(self):
        q = mbqueue.Queue()
        result = []

        def producer():
            time.sleep(0.05)
            q.put(42)

        t = threading.Thread(target=producer)
        t.start()
        item = q.get()  # should block until producer adds item
        result.append(item)
        t.join()
        assert result == [42]


# ---------------------------------------------------------------------------
# Shutdown (Python 3.13+ only)
# ---------------------------------------------------------------------------


class TestShutdown:
    def test_shutdown_exists(self):
        _skip_unless_shutdown()
        q = mbqueue.Queue()
        assert hasattr(q, "shutdown")

    def test_graceful_put_raises(self):
        _skip_unless_shutdown()
        q = mbqueue.Queue()
        q.shutdown()
        with pytest.raises(mbqueue.ShutDown):
            q.put(1)

    def test_graceful_get_drains(self):
        _skip_unless_shutdown()
        q = mbqueue.Queue()
        q.put(1)
        q.put(2)
        q.shutdown()
        assert q.get() == 1
        assert q.get() == 2
        with pytest.raises(mbqueue.ShutDown):
            q.get()

    def test_immediate_clears(self):
        _skip_unless_shutdown()
        q = mbqueue.Queue()
        q.put(1)
        q.put(2)
        q.shutdown(immediate=True)
        assert q.qsize() == 0
        with pytest.raises(mbqueue.ShutDown):
            q.get()

    def test_immediate_put_raises(self):
        _skip_unless_shutdown()
        q = mbqueue.Queue()
        q.shutdown(immediate=True)
        with pytest.raises(mbqueue.ShutDown):
            q.put(1)

    def test_shutdown_wakes_blocked_putter(self):
        _skip_unless_shutdown()
        q = mbqueue.Queue(maxsize=1)
        q.put("fill")
        exc = []

        def putter():
            try:
                q.put("blocked")
            except mbqueue.ShutDown:
                exc.append(True)

        t = threading.Thread(target=putter)
        t.start()
        time.sleep(0.05)
        q.shutdown()
        t.join(timeout=2)
        assert not t.is_alive()
        assert exc == [True]

    def test_shutdown_wakes_blocked_getter(self):
        _skip_unless_shutdown()
        q = mbqueue.Queue()
        exc = []

        def getter():
            try:
                q.get()
            except mbqueue.ShutDown:
                exc.append(True)

        t = threading.Thread(target=getter)
        t.start()
        time.sleep(0.05)
        q.shutdown()
        t.join(timeout=2)
        assert not t.is_alive()
        assert exc == [True]

    def test_immediate_shutdown_join(self):
        """immediate shutdown with items clears unfinished_tasks, so join returns."""
        _skip_unless_shutdown()
        q = mbqueue.Queue()
        q.put(1)
        q.put(2)
        # unfinished_tasks = 2
        q.shutdown(immediate=True)
        # items drained, unfinished_tasks should be 0
        q.join()  # should not block


# ---------------------------------------------------------------------------
# Concurrent stress
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_many_producers_consumers(self):
        q = mbqueue.Queue(maxsize=10)
        n_items = 200
        n_producers = 4
        n_consumers = 4
        items_per_producer = n_items // n_producers
        results = []
        lock = threading.Lock()

        def producer(start):
            for i in range(start, start + items_per_producer):
                q.put(i)

        def consumer(count):
            local = []
            for _ in range(count):
                local.append(q.get())
                q.task_done()
            with lock:
                results.extend(local)

        producers = [
            threading.Thread(target=producer, args=(i * items_per_producer,))
            for i in range(n_producers)
        ]
        items_per_consumer = n_items // n_consumers
        consumers = [
            threading.Thread(target=consumer, args=(items_per_consumer,))
            for _ in range(n_consumers)
        ]

        for t in producers + consumers:
            t.start()
        for t in producers + consumers:
            t.join(timeout=10)

        q.join()
        assert sorted(results) == list(range(n_items))

    def test_concurrent_put_get(self):
        """Rapidly alternate put/get from multiple threads."""
        q = mbqueue.Queue()
        n = 500
        got = []
        lock = threading.Lock()

        def putter():
            for i in range(n):
                q.put(i)

        def getter():
            local = []
            for _ in range(n):
                local.append(q.get())
                q.task_done()
            with lock:
                got.extend(local)

        tp = threading.Thread(target=putter)
        tg = threading.Thread(target=getter)
        tp.start()
        tg.start()
        tp.join(timeout=10)
        tg.join(timeout=10)
        q.join()
        assert len(got) == n


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_maxsize_one(self):
        q = mbqueue.Queue(maxsize=1)
        q.put("only")
        assert q.full()
        assert q.get() == "only"
        assert q.empty()

    def test_put_none(self):
        q = mbqueue.Queue()
        q.put(None)
        assert q.get() is None

    def test_exception_types(self):
        assert issubclass(mbqueue.Empty, Exception)
        assert issubclass(mbqueue.Full, Exception)
        if HAS_SHUTDOWN:
            assert issubclass(mbqueue.ShutDown, Exception)
