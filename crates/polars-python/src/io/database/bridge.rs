//! The Python driver bridge: database backends implemented by Python packages (ConnectorX,
//! DB-API drivers, SQLAlchemy, ODBC modules, connection factories and live connections),
//! behind the Rust backend traits.
//!
//! The Python side is `polars.io.database._bridge`. Sessions return Polars DataFrames, so every
//! driver the eager `read_database` supports is reachable.

use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_error::{PolarsResult, polars_err};
use polars_io::database::backend::{
    BackendKind, BatchIter, CancelHandle, DatabaseSession, ProbeResult, PythonDriverBridge,
};
use polars_io::database::connection::{ConnectionSpec, ConnectionTarget};
use polars_io::database::dialect::DialectName;
use polars_plan::plans::python_df_to_rust;
use polars_utils::pl_str::PlSmallStr;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;

fn bridge_module(py: Python<'_>) -> PyResult<&Bound<'_, PyModule>> {
    static MODULE: PyOnceLock<Py<PyModule>> = PyOnceLock::new();
    MODULE
        .get_or_try_init(py, || Ok(py.import("polars.io.database._bridge")?.unbind()))
        .map(|m| m.bind(py))
}

fn err(e: PyErr) -> polars_error::PolarsError {
    polars_err!(ComputeError: "{e}")
}

/// The target of a connection as the Python side expects it: `(kind, object)`.
fn target<'py>(
    py: Python<'py>,
    spec: &ConnectionSpec,
) -> PyResult<(&'static str, Bound<'py, PyAny>)> {
    Ok(match &spec.target {
        ConnectionTarget::Uri(uri) => ("uri", uri.as_str().into_pyobject(py)?.into_any()),
        ConnectionTarget::Odbc(s) => ("odbc", s.as_str().into_pyobject(py)?.into_any()),
        ConnectionTarget::Factory(f) => ("factory", f.0.bind(py).clone()),
        ConnectionTarget::Live(c) => ("live", c.0.0.bind(py).clone()),
    })
}

fn options(opts: &[(PlSmallStr, PlSmallStr)]) -> Vec<(&str, &str)> {
    opts.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect()
}

pub struct PyDriverBridge;

impl PyDriverBridge {
    fn connection_info(&self, spec: &ConnectionSpec) -> PolarsResult<(DialectName, PlSmallStr)> {
        Python::attach(|py| {
            let (kind, obj) = target(py, spec)?;
            let (dialect, module): (String, String) = bridge_module(py)?
                .call_method1("connection_info", (obj, kind))?
                .extract()?;
            PyResult::Ok((dialect, module))
        })
        .map_err(err)
        .map(|(dialect, module)| {
            (
                DialectName::parse(&dialect)
                    .unwrap_or_else(|_| DialectName::from_python_module(&module)),
                module.into(),
            )
        })
    }
}

impl PythonDriverBridge for PyDriverBridge {
    fn probe(&self, kind: &BackendKind) -> Option<ProbeResult> {
        Python::attach(|py| {
            let out = bridge_module(py)
                .ok()?
                .call_method1("probe", (kind.to_string(),))
                .ok()?;
            if out.is_none() {
                return None;
            }
            let (version, location): (Option<String>, Option<String>) = out.extract().ok()?;
            Some(ProbeResult {
                version: version.map(Into::into),
                location: location.map(Into::into),
            })
        })
    }

    fn connection_dialect(&self, spec: &ConnectionSpec) -> PolarsResult<DialectName> {
        Ok(self.connection_info(spec)?.0)
    }

    fn connection_module(&self, spec: &ConnectionSpec) -> PolarsResult<PlSmallStr> {
        Ok(self.connection_info(spec)?.1)
    }

    fn connect(
        &self,
        kind: &BackendKind,
        spec: &ConnectionSpec,
        dialect: DialectName,
        batch_size: Option<usize>,
    ) -> PolarsResult<Box<dyn DatabaseSession>> {
        Python::attach(|py| {
            let (target_kind, obj) = target(py, spec)?;
            let session = bridge_module(py)?.call_method1(
                "connect",
                (
                    kind.to_string(),
                    target_kind,
                    obj,
                    options(&spec.connect_options),
                    options(&spec.execute_options),
                    dialect.to_string(),
                    spec.credential_provider
                        .as_ref()
                        .map(|p| p.0.bind(py).clone()),
                    spec.query_params.as_ref().map(|p| p.0.bind(py).clone()),
                    spec.python_options.as_ref().map(|p| p.0.bind(py).clone()),
                ),
            )?;
            let can_cancel: bool = session.getattr("can_cancel")?.extract()?;
            PyResult::Ok(Box::new(PySession {
                session: Arc::new(session.unbind()),
                can_cancel,
                batch_size,
            }) as Box<dyn DatabaseSession>)
        })
        .map_err(err)
    }

    fn credentials(
        &self,
        spec: &ConnectionSpec,
    ) -> PolarsResult<Option<(Option<String>, Option<String>)>> {
        let Some(provider) = &spec.credential_provider else {
            return Ok(None);
        };
        Python::attach(|py| {
            bridge_module(py)?
                .call_method1("credentials", (provider.0.bind(py),))?
                .extract::<Option<(Option<String>, Option<String>)>>()
        })
        .map_err(err)
    }
}

/// A session of a Python backend.
struct PySession {
    session: Arc<Py<PyAny>>,
    can_cancel: bool,
    batch_size: Option<usize>,
}

impl DatabaseSession for PySession {
    fn execute(&mut self, sql: &str, batch_size: Option<usize>) -> PolarsResult<BatchIter> {
        let iter = Python::attach(|py| {
            self.session
                .call_method1(py, "execute", (sql, batch_size.or(self.batch_size)))
        })
        .map_err(err)?;
        Ok(Box::new(PyBatches { iter: Some(iter) }))
    }

    fn execute_update(&mut self, sql: &str) -> PolarsResult<()> {
        Python::attach(|py| self.session.call_method1(py, "execute_update", (sql,)))
            .map_err(err)?;
        Ok(())
    }

    fn cancel_handle(&self) -> Option<CancelHandle> {
        if !self.can_cancel {
            return None;
        }
        let session = self.session.clone();
        Some(Arc::new(move || {
            Python::attach(|py| session.call_method0(py, "cancel"))
                .map_err(err)
                .map(|_| ())
        }))
    }
}

impl Drop for PySession {
    fn drop(&mut self) {
        Python::attach(|py| {
            if let Err(e) = self.session.call_method0(py, "close") {
                if polars_core::config::verbose() {
                    eprintln!("[scan_database]: closing a Python session failed: {e}");
                }
            }
        });
    }
}

/// Batches from a Python iterator of DataFrames.
struct PyBatches {
    iter: Option<Py<PyAny>>,
}

impl Iterator for PyBatches {
    type Item = PolarsResult<DataFrame>;

    fn next(&mut self) -> Option<Self::Item> {
        let iter = self.iter.as_ref()?;
        let out = Python::attach(|py| {
            let builtins = py.import("builtins")?;
            let sentinel = py.None();
            let df = builtins.call_method1("next", (iter.bind(py), &sentinel))?;
            if df.is_none() {
                return PyResult::Ok(None);
            }
            Ok(Some(python_df_to_rust(py, df)))
        });
        match out {
            Ok(Some(df)) => Some(df),
            Ok(None) => {
                self.iter = None;
                None
            },
            Err(e) => {
                self.iter = None;
                Some(Err(err(e)))
            },
        }
    }
}

impl Drop for PyBatches {
    fn drop(&mut self) {
        if let Some(iter) = self.iter.take() {
            Python::attach(|py| drop(iter.into_bound(py)));
        }
    }
}
