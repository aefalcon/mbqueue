use std::collections::VecDeque;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex, MutexGuard};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;

// ---------------------------------------------------------------------------
// Cached exception types from the `queue` stdlib module
// ---------------------------------------------------------------------------

static QUEUE_EMPTY: OnceLock<Py<PyType>> = OnceLock::new();
static QUEUE_FULL: OnceLock<Py<PyType>> = OnceLock::new();
#[cfg(Py_3_13)]
static QUEUE_SHUTDOWN: OnceLock<Py<PyType>> = OnceLock::new();

fn empty_err() -> PyErr {
    Python::attach(|py| PyErr::from_type(QUEUE_EMPTY.get().unwrap().bind(py).clone(), ""))
}

fn full_err() -> PyErr {
    Python::attach(|py| PyErr::from_type(QUEUE_FULL.get().unwrap().bind(py).clone(), ""))
}

#[cfg(Py_3_13)]
fn shutdown_err() -> PyErr {
    Python::attach(|py| {
        PyErr::from_type(
            QUEUE_SHUTDOWN.get().unwrap().bind(py).clone(),
            "the queue has been shut down",
        )
    })
}

// ---------------------------------------------------------------------------
// Internal error type (used inside detach closures)
// ---------------------------------------------------------------------------

enum QueueError {
    Empty,
    Full,
    #[cfg(Py_3_13)]
    ShutDown,
}

impl QueueError {
    fn into_pyerr(self) -> PyErr {
        match self {
            QueueError::Empty => empty_err(),
            QueueError::Full => full_err(),
            #[cfg(Py_3_13)]
            QueueError::ShutDown => shutdown_err(),
        }
    }
}

// ---------------------------------------------------------------------------
// QueueInner
// ---------------------------------------------------------------------------

struct QueueInner {
    items: VecDeque<Py<PyAny>>,
    effective_maxsize: usize, // 0 = unbounded
    unfinished_tasks: usize,
    not_empty_waiters: usize,
    not_full_waiters: usize,
    all_tasks_done_waiters: usize,
    #[cfg(Py_3_13)]
    is_shutdown: bool,
}

impl QueueInner {
    fn new(maxsize: isize) -> Self {
        Self {
            items: VecDeque::new(),
            effective_maxsize: if maxsize <= 0 { 0 } else { maxsize as usize },
            unfinished_tasks: 0,
            not_empty_waiters: 0,
            not_full_waiters: 0,
            all_tasks_done_waiters: 0,
            #[cfg(Py_3_13)]
            is_shutdown: false,
        }
    }

    fn is_full(&self) -> bool {
        self.effective_maxsize > 0 && self.items.len() >= self.effective_maxsize
    }
}

// ---------------------------------------------------------------------------
// Queue pyclass
// ---------------------------------------------------------------------------

#[pyclass(frozen)]
struct Queue {
    raw_maxsize: isize,
    inner: Mutex<QueueInner>,
    not_empty: Condvar,
    not_full: Condvar,
    all_tasks_done: Condvar,
}

/// Parse timeout: None => None, negative => None (treated as infinite), positive => Some(Duration)
fn parse_timeout(timeout: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Duration>> {
    match timeout {
        None => Ok(None),
        Some(obj) => {
            if obj.is_none() {
                return Ok(None);
            }
            let secs: f64 = obj.extract()?;
            if secs < 0.0 {
                Ok(None)
            } else {
                Ok(Some(Duration::from_secs_f64(secs)))
            }
        }
    }
}

/// Helper: execute the put logic on an already-locked guard.
enum PutFast {
    Done,
    NeedBlock(Py<PyAny>),
}

fn try_put_fast(
    inner: &mut MutexGuard<'_, QueueInner>,
    item: Py<PyAny>,
    block: bool,
    not_empty: &Condvar,
) -> Result<PutFast, PyErr> {
    #[cfg(Py_3_13)]
    if inner.is_shutdown {
        return Err(shutdown_err());
    }
    if !inner.is_full() {
        inner.items.push_back(item);
        inner.unfinished_tasks += 1;
        if inner.not_empty_waiters > 0 {
            not_empty.notify_one();
        }
        return Ok(PutFast::Done);
    }
    if !block {
        return Err(full_err());
    }
    Ok(PutFast::NeedBlock(item))
}

/// Helper: execute the get logic on an already-locked guard.
enum GetFast {
    Done(Py<PyAny>),
    NeedBlock,
}

fn try_get_fast(
    inner: &mut MutexGuard<'_, QueueInner>,
    block: bool,
    not_full: &Condvar,
) -> Result<GetFast, PyErr> {
    if let Some(item) = inner.items.pop_front() {
        if inner.not_full_waiters > 0 {
            not_full.notify_one();
        }
        return Ok(GetFast::Done(item));
    }
    #[cfg(Py_3_13)]
    if inner.is_shutdown {
        return Err(shutdown_err());
    }
    if !block {
        return Err(empty_err());
    }
    Ok(GetFast::NeedBlock)
}

#[pymethods]
impl Queue {
    #[new]
    #[pyo3(signature = (maxsize=0))]
    fn new(maxsize: isize) -> Self {
        Queue {
            raw_maxsize: maxsize,
            inner: Mutex::new(QueueInner::new(maxsize)),
            not_empty: Condvar::new(),
            not_full: Condvar::new(),
            all_tasks_done: Condvar::new(),
        }
    }

    #[getter]
    fn maxsize(&self) -> isize {
        self.raw_maxsize
    }

    fn qsize(&self) -> usize {
        self.inner.lock().items.len()
    }

    fn empty(&self) -> bool {
        self.inner.lock().items.is_empty()
    }

    fn full(&self) -> bool {
        self.inner.lock().is_full()
    }

    #[pyo3(signature = (item, block=true, timeout=None))]
    fn put(
        &self,
        py: Python<'_>,
        item: Py<PyAny>,
        block: bool,
        timeout: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        // Fast path: try-lock without detaching from the interpreter.
        if let Some(mut inner) = self.inner.try_lock() {
            match try_put_fast(&mut inner, item, block, &self.not_empty)? {
                PutFast::Done => return Ok(()),
                PutFast::NeedBlock(item_back) => {
                    // Need to block — drop lock, fall through to slow path.
                    drop(inner);
                    return self.put_slow(py, item_back, timeout);
                }
            }
        }

        // Contended — detach and lock.
        self.put_slow(py, item, timeout)
    }

    fn put_nowait(&self, py: Python<'_>, item: Py<PyAny>) -> PyResult<()> {
        // Fast path: try-lock without detaching.
        if let Some(mut inner) = self.inner.try_lock() {
            return match try_put_fast(&mut inner, item, false, &self.not_empty) {
                Ok(PutFast::Done) => Ok(()),
                Ok(PutFast::NeedBlock(_)) => unreachable!(),
                Err(e) => Err(e),
            };
        }

        // Contended — fall back to detach + lock.
        let result = py.detach(|| {
            let mut inner = self.inner.lock();

            #[cfg(Py_3_13)]
            if inner.is_shutdown {
                return Err(QueueError::ShutDown);
            }

            if inner.is_full() {
                return Err(QueueError::Full);
            }

            inner.items.push_back(item);
            inner.unfinished_tasks += 1;
            if inner.not_empty_waiters > 0 {
                self.not_empty.notify_one();
            }
            Ok(())
        });

        result.map_err(|e| e.into_pyerr())
    }

    #[pyo3(signature = (block=true, timeout=None))]
    fn get(
        &self,
        py: Python<'_>,
        block: bool,
        timeout: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        // Fast path: try-lock without detaching.
        if let Some(mut inner) = self.inner.try_lock() {
            match try_get_fast(&mut inner, block, &self.not_full)? {
                GetFast::Done(item) => return Ok(item),
                GetFast::NeedBlock => {
                    drop(inner);
                    return self.get_slow(py, timeout);
                }
            }
        }

        // Contended — detach and lock.
        self.get_slow(py, timeout)
    }

    fn get_nowait(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // Fast path: try-lock without detaching.
        if let Some(mut inner) = self.inner.try_lock() {
            return match try_get_fast(&mut inner, false, &self.not_full) {
                Ok(GetFast::Done(item)) => Ok(item),
                Ok(GetFast::NeedBlock) => unreachable!(),
                Err(e) => Err(e),
            };
        }

        // Contended — fall back to detach + lock.
        let result: Result<Py<PyAny>, QueueError> = py.detach(|| {
            let mut inner = self.inner.lock();

            if let Some(item) = inner.items.pop_front() {
                if inner.not_full_waiters > 0 {
                    self.not_full.notify_one();
                }
                return Ok(item);
            }

            #[cfg(Py_3_13)]
            if inner.is_shutdown {
                return Err(QueueError::ShutDown);
            }

            Err(QueueError::Empty)
        });

        result.map_err(|e| e.into_pyerr())
    }

    fn task_done(&self, py: Python<'_>) -> PyResult<()> {
        if let Some(mut inner) = self.inner.try_lock() {
            if inner.unfinished_tasks == 0 {
                return Err(PyValueError::new_err("task_done() called too many times"));
            }
            inner.unfinished_tasks -= 1;
            if inner.unfinished_tasks == 0 && inner.all_tasks_done_waiters > 0 {
                self.all_tasks_done.notify_all();
            }
            return Ok(());
        }

        py.detach(|| {
            let mut inner = self.inner.lock();
            if inner.unfinished_tasks == 0 {
                return Err(PyValueError::new_err("task_done() called too many times"));
            }
            inner.unfinished_tasks -= 1;
            if inner.unfinished_tasks == 0 && inner.all_tasks_done_waiters > 0 {
                self.all_tasks_done.notify_all();
            }
            Ok(())
        })
    }

    fn join(&self, py: Python<'_>) {
        py.detach(|| {
            let mut inner = self.inner.lock();
            while inner.unfinished_tasks > 0 {
                inner.all_tasks_done_waiters += 1;
                self.all_tasks_done.wait(&mut inner);
                inner.all_tasks_done_waiters -= 1;
            }
        });
    }

    #[cfg(Py_3_13)]
    #[pyo3(signature = (immediate=false))]
    fn shutdown(&self, py: Python<'_>, immediate: bool) {
        py.detach(|| {
            let mut inner = self.inner.lock();
            inner.is_shutdown = true;

            if immediate {
                let drained = inner.items.len();
                inner.items.clear();
                inner.unfinished_tasks = inner.unfinished_tasks.saturating_sub(drained);
                if inner.unfinished_tasks == 0 && inner.all_tasks_done_waiters > 0 {
                    self.all_tasks_done.notify_all();
                }
            }

            if inner.not_full_waiters > 0 {
                self.not_full.notify_all();
            }
            if inner.not_empty_waiters > 0 {
                self.not_empty.notify_all();
            }
        });
    }

    fn __traverse__(&self, visit: pyo3::gc::PyVisit<'_>) -> Result<(), pyo3::gc::PyTraverseError> {
        let inner = self.inner.lock();
        for item in &inner.items {
            visit.call(item)?;
        }
        Ok(())
    }

    fn __clear__(&self) {
        self.inner.lock().items.clear();
    }
}

// Private slow-path methods (not exposed to Python).
impl Queue {
    fn put_slow(
        &self,
        py: Python<'_>,
        item: Py<PyAny>,
        timeout: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let deadline = parse_timeout(timeout)?.map(|d| Instant::now() + d);

        let result = py.detach(|| {
            let mut inner = self.inner.lock();

            loop {
                #[cfg(Py_3_13)]
                if inner.is_shutdown {
                    return Err(QueueError::ShutDown);
                }

                if !inner.is_full() {
                    break;
                }

                inner.not_full_waiters += 1;
                match deadline {
                    Some(dl) => {
                        let now = Instant::now();
                        if now >= dl {
                            inner.not_full_waiters -= 1;
                            return Err(QueueError::Full);
                        }
                        let timed_out = self.not_full.wait_for(&mut inner, dl - now).timed_out();
                        inner.not_full_waiters -= 1;
                        if timed_out && inner.is_full() {
                            return Err(QueueError::Full);
                        }
                    }
                    None => {
                        self.not_full.wait(&mut inner);
                        inner.not_full_waiters -= 1;
                    }
                }
            }

            inner.items.push_back(item);
            inner.unfinished_tasks += 1;
            if inner.not_empty_waiters > 0 {
                self.not_empty.notify_one();
            }
            Ok(())
        });

        result.map_err(|e| e.into_pyerr())
    }

    fn get_slow(&self, py: Python<'_>, timeout: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        let deadline = parse_timeout(timeout)?.map(|d| Instant::now() + d);

        let result: Result<Py<PyAny>, QueueError> = py.detach(|| {
            let mut inner = self.inner.lock();

            loop {
                if let Some(item) = inner.items.pop_front() {
                    if inner.not_full_waiters > 0 {
                        self.not_full.notify_one();
                    }
                    return Ok(item);
                }

                #[cfg(Py_3_13)]
                if inner.is_shutdown {
                    return Err(QueueError::ShutDown);
                }

                inner.not_empty_waiters += 1;
                match deadline {
                    Some(dl) => {
                        let now = Instant::now();
                        if now >= dl {
                            inner.not_empty_waiters -= 1;
                            return Err(QueueError::Empty);
                        }
                        let timed_out = self.not_empty.wait_for(&mut inner, dl - now).timed_out();
                        inner.not_empty_waiters -= 1;
                        if timed_out && inner.items.is_empty() {
                            #[cfg(Py_3_13)]
                            if inner.is_shutdown {
                                return Err(QueueError::ShutDown);
                            }
                            return Err(QueueError::Empty);
                        }
                    }
                    None => {
                        self.not_empty.wait(&mut inner);
                        inner.not_empty_waiters -= 1;
                    }
                }
            }
        });

        result.map_err(|e| e.into_pyerr())
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// Import a type from a Python module and return it as `Py<PyType>`.
fn import_type(py: Python<'_>, module: &str, name: &str) -> PyResult<Py<PyType>> {
    let ty = py.import(module)?.getattr(name)?;
    Ok(ty.cast_into::<PyType>()?.unbind())
}

#[pymodule(gil_used = false)]
fn mbqueue(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Cache stdlib exception types.
    QUEUE_EMPTY.set(import_type(py, "queue", "Empty")?).unwrap();
    QUEUE_FULL.set(import_type(py, "queue", "Full")?).unwrap();
    #[cfg(Py_3_13)]
    QUEUE_SHUTDOWN
        .set(import_type(py, "queue", "ShutDown")?)
        .unwrap();

    m.add_class::<Queue>()?;

    // Re-export stdlib exception types as mbqueue.Empty, mbqueue.Full, etc.
    m.add("Empty", QUEUE_EMPTY.get().unwrap().bind(py))?;
    m.add("Full", QUEUE_FULL.get().unwrap().bind(py))?;
    #[cfg(Py_3_13)]
    m.add("ShutDown", QUEUE_SHUTDOWN.get().unwrap().bind(py))?;

    Ok(())
}
