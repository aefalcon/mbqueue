# mbqueue

A performance experiment: `queue.Queue` reimplemented as a Rust extension
using [PyO3](https://pyo3.rs/) and
[parking_lot](https://docs.rs/parking_lot/).

This is a drop-in replacement for Python's `queue.Queue`. It re-exports
`queue.Empty`, `queue.Full`, and `queue.ShutDown` so existing `except`
clauses work unchanged.

```python
import mbqueue

q = mbqueue.Queue(maxsize=10)
q.put("hello")
assert q.get() == "hello"
```

## Why

Free-threaded Python (3.13t/3.14t) removes the GIL, which means lock
overhead in the pure-Python `queue.Queue` implementation actually matters.
This project explores how much a native-extension queue can improve on that.

The answer: **3-4x faster in synthetic benchmarks**, but `queue.Queue`
already handles ~225k items/sec per thread. For most applications the queue
is not the bottleneck.

## Benchmark

10,000 items per round, unbounded queue, measured on Python 3.14t
(free-threaded) with `pytest-benchmark`:

```
Name (time in ms)         Median               Speedup
------------------------------------------------------
test_spsc_mbqueue          5.7
test_spsc_stdlib          21.3                  3.7x
test_mpsc_mbqueue          8.2
test_mpsc_stdlib          29.5                  3.6x
test_spmc_mbqueue         11.7
test_spmc_stdlib          44.5                  3.8x
```

Run them yourself:

```bash
uv venv --python 3.14t
uv sync --group dev
uv run maturin develop
uv run python -m pytest tests/bench_queue.py -v
```

## Implementation notes

- `#[pyclass(frozen)]` with interior mutability via a single
  `parking_lot::Mutex` and three `Condvar`s (`not_empty`, `not_full`,
  `all_tasks_done`), mirroring CPython's design.
- `py.detach()` releases the interpreter before blocking on a condvar.
- Try-lock fast path avoids interpreter detach on uncontended operations.
- `shutdown()` is available on Python 3.13+ (`#[cfg(Py_3_13)]`).
- GC integration via `__traverse__` / `__clear__`.
