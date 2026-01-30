import queue
import threading


import mbqueue

N = 10_000
SENTINEL = None


# ---------------------------------------------------------------------------
# SPSC: 1 producer, 1 consumer
# ---------------------------------------------------------------------------


def _spsc_roundtrip(q_cls, n):
    q = q_cls()

    def producer():
        for i in range(n):
            q.put(i)
        q.put(SENTINEL)

    def consumer():
        while True:
            if q.get() is SENTINEL:
                break

    tp = threading.Thread(target=producer)
    tc = threading.Thread(target=consumer)
    tp.start()
    tc.start()
    tp.join()
    tc.join()


def test_spsc_mbqueue(benchmark):
    benchmark(_spsc_roundtrip, mbqueue.Queue, N)


def test_spsc_stdlib(benchmark):
    benchmark(_spsc_roundtrip, queue.Queue, N)


# ---------------------------------------------------------------------------
# MPSC: 4 producers, 1 consumer
# ---------------------------------------------------------------------------

N_PRODUCERS = 4


def _mpsc_roundtrip(q_cls, n_per_producer, n_producers):
    q = q_cls()
    total = n_per_producer * n_producers

    def producer():
        for i in range(n_per_producer):
            q.put(i)

    def consumer():
        for _ in range(total):
            q.get()

    producers = [threading.Thread(target=producer) for _ in range(n_producers)]
    tc = threading.Thread(target=consumer)
    for t in producers:
        t.start()
    tc.start()
    for t in producers:
        t.join()
    tc.join()


def test_mpsc_mbqueue(benchmark):
    benchmark(_mpsc_roundtrip, mbqueue.Queue, N // N_PRODUCERS, N_PRODUCERS)


def test_mpsc_stdlib(benchmark):
    benchmark(_mpsc_roundtrip, queue.Queue, N // N_PRODUCERS, N_PRODUCERS)


# ---------------------------------------------------------------------------
# SPMC: 1 producer, 4 consumers
# ---------------------------------------------------------------------------

N_CONSUMERS = 4


def _spmc_roundtrip(q_cls, n, n_consumers):
    q = q_cls()

    def producer():
        for i in range(n):
            q.put(i)
        for _ in range(n_consumers):
            q.put(SENTINEL)

    def consumer():
        while True:
            if q.get() is SENTINEL:
                break

    tp = threading.Thread(target=producer)
    consumers = [threading.Thread(target=consumer) for _ in range(n_consumers)]
    tp.start()
    for t in consumers:
        t.start()
    tp.join()
    for t in consumers:
        t.join()


def test_spmc_mbqueue(benchmark):
    benchmark(_spmc_roundtrip, mbqueue.Queue, N, N_CONSUMERS)


def test_spmc_stdlib(benchmark):
    benchmark(_spmc_roundtrip, queue.Queue, N, N_CONSUMERS)
