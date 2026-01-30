use std::collections::VecDeque;
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Exception classes
// ---------------------------------------------------------------------------

pyo3::create_exception!(mbqueue, Empty, pyo3::exceptions::PyException);
pyo3::create_exception!(mbqueue, Full, pyo3::exceptions::PyException);

#[cfg(Py_3_13)]
pyo3::create_exception!(mbqueue, ShutDown, pyo3::exceptions::PyException);

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
            QueueError::Empty => Empty::new_err(""),
            QueueError::Full => Full::new_err(""),
            #[cfg(Py_3_13)]
            QueueError::ShutDown => ShutDown::new_err("the queue has been shut down"),
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
    #[cfg(Py_3_13)]
    is_shutdown_immediate: bool,
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
            #[cfg(Py_3_13)]
            is_shutdown_immediate: false,
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
        let inner = self.inner.lock().unwrap();
        inner.items.len()
    }

    fn empty(&self) -> bool {
        let inner = self.inner.lock().unwrap();
        inner.items.is_empty()
    }

    fn full(&self) -> bool {
        let inner = self.inner.lock().unwrap();
        inner.is_full()
    }

    #[pyo3(signature = (item, block=true, timeout=None))]
    fn put(
        &self,
        py: Python<'_>,
        item: Py<PyAny>,
        block: bool,
        timeout: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let deadline = if block {
            parse_timeout(timeout)?.map(|d| Instant::now() + d)
        } else {
            None
        };

        let result = py.detach(|| {
            let mut inner = self.inner.lock().unwrap();

            loop {
                #[cfg(Py_3_13)]
                if inner.is_shutdown {
                    return Err(QueueError::ShutDown);
                }

                if !inner.is_full() {
                    break;
                }

                if !block {
                    return Err(QueueError::Full);
                }

                inner.not_full_waiters += 1;
                match deadline {
                    Some(dl) => {
                        let now = Instant::now();
                        if now >= dl {
                            inner.not_full_waiters -= 1;
                            return Err(QueueError::Full);
                        }
                        let (guard, wait_result) =
                            self.not_full.wait_timeout(inner, dl - now).unwrap();
                        inner = guard;
                        inner.not_full_waiters -= 1;
                        if wait_result.timed_out() && inner.is_full() {
                            return Err(QueueError::Full);
                        }
                    }
                    None => {
                        inner = self.not_full.wait(inner).unwrap();
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

    fn put_nowait(&self, py: Python<'_>, item: Py<PyAny>) -> PyResult<()> {
        let result = py.detach(|| {
            let mut inner = self.inner.lock().unwrap();

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
        let deadline = if block {
            parse_timeout(timeout)?.map(|d| Instant::now() + d)
        } else {
            None
        };

        let result: Result<Py<PyAny>, QueueError> = py.detach(|| {
            let mut inner = self.inner.lock().unwrap();

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

                if !block {
                    return Err(QueueError::Empty);
                }

                inner.not_empty_waiters += 1;
                match deadline {
                    Some(dl) => {
                        let now = Instant::now();
                        if now >= dl {
                            inner.not_empty_waiters -= 1;
                            return Err(QueueError::Empty);
                        }
                        let (guard, wait_result) =
                            self.not_empty.wait_timeout(inner, dl - now).unwrap();
                        inner = guard;
                        inner.not_empty_waiters -= 1;
                        if wait_result.timed_out() && inner.items.is_empty() {
                            #[cfg(Py_3_13)]
                            if inner.is_shutdown {
                                return Err(QueueError::ShutDown);
                            }
                            return Err(QueueError::Empty);
                        }
                    }
                    None => {
                        inner = self.not_empty.wait(inner).unwrap();
                        inner.not_empty_waiters -= 1;
                    }
                }
            }
        });

        result.map_err(|e| e.into_pyerr())
    }

    fn get_nowait(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let result: Result<Py<PyAny>, QueueError> = py.detach(|| {
            let mut inner = self.inner.lock().unwrap();

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
        py.detach(|| {
            let mut inner = self.inner.lock().unwrap();
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
            let mut inner = self.inner.lock().unwrap();
            while inner.unfinished_tasks > 0 {
                inner.all_tasks_done_waiters += 1;
                inner = self.all_tasks_done.wait(inner).unwrap();
                inner.all_tasks_done_waiters -= 1;
            }
        });
    }

    #[cfg(Py_3_13)]
    #[pyo3(signature = (immediate=false))]
    fn shutdown(&self, py: Python<'_>, immediate: bool) {
        py.detach(|| {
            let mut inner = self.inner.lock().unwrap();
            inner.is_shutdown = true;

            if immediate {
                inner.is_shutdown_immediate = true;
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
        let inner = self.inner.lock().unwrap();
        for item in &inner.items {
            visit.call(item)?;
        }
        Ok(())
    }

    fn __clear__(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.items.clear();
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pymodule(gil_used = false)]
fn mbqueue(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Queue>()?;
    m.add("Empty", m.py().get_type::<Empty>())?;
    m.add("Full", m.py().get_type::<Full>())?;
    #[cfg(Py_3_13)]
    m.add("ShutDown", m.py().get_type::<ShutDown>())?;
    Ok(())
}
