use std::collections::VecDeque;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;

const SIGNAL_POLL_INTERVAL: Duration = Duration::from_millis(100);

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
    Interrupted(PyErr),
}

impl QueueError {
    fn into_pyerr(self) -> PyErr {
        match self {
            QueueError::Empty => empty_err(),
            QueueError::Full => full_err(),
            #[cfg(Py_3_13)]
            QueueError::ShutDown => shutdown_err(),
            QueueError::Interrupted(e) => e,
        }
    }
}

fn check_signals() -> Result<(), QueueError> {
    Python::attach(|py| py.check_signals()).map_err(QueueError::Interrupted)
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

    /// Decrement unfinished_tasks. Returns `true` if joiners should be notified.
    fn complete_task(&mut self) -> PyResult<bool> {
        if self.unfinished_tasks == 0 {
            return Err(PyValueError::new_err("task_done() called too many times"));
        }
        self.unfinished_tasks -= 1;
        Ok(self.unfinished_tasks == 0 && self.all_tasks_done_waiters > 0)
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

/// Parse timeout: None/PyNone/negative => None (infinite), positive => Some(Duration).
fn parse_timeout(timeout: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Duration>> {
    let Some(obj) = timeout else { return Ok(None) };
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

/// Attempt to put an item. Returns `Full(item)` if the queue is full.
enum PutOutcome {
    Done,
    Full(Py<PyAny>),
}

fn try_put(
    inner: &mut QueueInner,
    item: Py<PyAny>,
    not_empty: &Condvar,
) -> Result<PutOutcome, QueueError> {
    #[cfg(Py_3_13)]
    if inner.is_shutdown {
        return Err(QueueError::ShutDown);
    }
    if !inner.is_full() {
        inner.items.push_back(item);
        inner.unfinished_tasks += 1;
        if inner.not_empty_waiters > 0 {
            not_empty.notify_one();
        }
        return Ok(PutOutcome::Done);
    }
    Ok(PutOutcome::Full(item))
}

/// Attempt to get an item. Returns `Empty` if the queue is empty.
enum GetOutcome {
    Done(Py<PyAny>),
    Empty,
}

fn try_get(inner: &mut QueueInner, not_full: &Condvar) -> Result<GetOutcome, QueueError> {
    if let Some(item) = inner.items.pop_front() {
        if inner.not_full_waiters > 0 {
            not_full.notify_one();
        }
        return Ok(GetOutcome::Done(item));
    }
    #[cfg(Py_3_13)]
    if inner.is_shutdown {
        return Err(QueueError::ShutDown);
    }
    Ok(GetOutcome::Empty)
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
            match try_put(&mut inner, item, &self.not_empty) {
                Ok(PutOutcome::Done) => return Ok(()),
                Ok(PutOutcome::Full(item_back)) => {
                    if !block {
                        return Err(full_err());
                    }
                    drop(inner);
                    return self.put_slow(py, item_back, timeout);
                }
                Err(e) => return Err(e.into_pyerr()),
            }
        }

        // Contended — detach and lock.
        if block {
            self.put_slow(py, item, timeout)
        } else {
            py.detach(|| {
                let mut inner = self.inner.lock();
                match try_put(&mut inner, item, &self.not_empty) {
                    Ok(PutOutcome::Done) => Ok(()),
                    Ok(PutOutcome::Full(_)) => Err(QueueError::Full),
                    Err(e) => Err(e),
                }
            })
            .map_err(|e| e.into_pyerr())
        }
    }

    fn put_nowait(&self, py: Python<'_>, item: Py<PyAny>) -> PyResult<()> {
        self.put(py, item, false, None)
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
            match try_get(&mut inner, &self.not_full) {
                Ok(GetOutcome::Done(item)) => return Ok(item),
                Ok(GetOutcome::Empty) => {
                    if !block {
                        return Err(empty_err());
                    }
                    drop(inner);
                    return self.get_slow(py, timeout);
                }
                Err(e) => return Err(e.into_pyerr()),
            }
        }

        // Contended — detach and lock.
        if block {
            self.get_slow(py, timeout)
        } else {
            py.detach(|| {
                let mut inner = self.inner.lock();
                match try_get(&mut inner, &self.not_full) {
                    Ok(GetOutcome::Done(item)) => Ok(item),
                    Ok(GetOutcome::Empty) => Err(QueueError::Empty),
                    Err(e) => Err(e),
                }
            })
            .map_err(|e| e.into_pyerr())
        }
    }

    fn get_nowait(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.get(py, false, None)
    }

    fn task_done(&self, py: Python<'_>) -> PyResult<()> {
        if let Some(mut inner) = self.inner.try_lock() {
            if inner.complete_task()? {
                self.all_tasks_done.notify_all();
            }
            return Ok(());
        }

        py.detach(|| {
            let mut inner = self.inner.lock();
            if inner.complete_task()? {
                self.all_tasks_done.notify_all();
            }
            Ok(())
        })
    }

    fn join(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            let mut inner = self.inner.lock();
            while inner.unfinished_tasks > 0 {
                inner.all_tasks_done_waiters += 1;
                self.all_tasks_done
                    .wait_for(&mut inner, SIGNAL_POLL_INTERVAL);
                inner.all_tasks_done_waiters -= 1;
                check_signals().map_err(|e| e.into_pyerr())?;
            }
            Ok(())
        })
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
                let wait_time = match deadline {
                    Some(dl) => {
                        let now = Instant::now();
                        if now >= dl {
                            inner.not_full_waiters -= 1;
                            return Err(QueueError::Full);
                        }
                        (dl - now).min(SIGNAL_POLL_INTERVAL)
                    }
                    None => SIGNAL_POLL_INTERVAL,
                };
                self.not_full.wait_for(&mut inner, wait_time);
                inner.not_full_waiters -= 1;
                check_signals()?;
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
                let wait_time = match deadline {
                    Some(dl) => {
                        let now = Instant::now();
                        if now >= dl {
                            inner.not_empty_waiters -= 1;
                            return Err(QueueError::Empty);
                        }
                        (dl - now).min(SIGNAL_POLL_INTERVAL)
                    }
                    None => SIGNAL_POLL_INTERVAL,
                };
                self.not_empty.wait_for(&mut inner, wait_time);
                inner.not_empty_waiters -= 1;
                check_signals()?;
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
