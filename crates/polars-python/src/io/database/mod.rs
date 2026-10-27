//! `scan_database`: argument conversion, the sqlglot renderer and the Python driver bridge.
//!
//! Everything else (resolution, SQL compilation, execution) is in Rust, in `polars-io`,
//! `polars-plan` and `polars-stream`.

mod bridge;
mod sqlglot;

use std::sync::{Arc, Once};

use polars::prelude::*;
use polars_io::database::backend::resolve::{candidate_statuses, install_hint, probe};
use polars_io::database::backend::{BackendKind, adbc, register_python_bridge};
use polars_io::database::connection::{ConnectionSpec, ConnectionTarget, LiveConnection};
use polars_io::database::sql::register_sql_renderer;
use polars_io::database::{
    Consistency, DatabaseRelation, DatabaseScanOptions, DialectName, FetchOptions, PartitionSpec,
    PartitionStrategy,
};
use polars_plan::dsl::DslBuilder;
use polars_utils::python_function::PythonObject;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::conversion::Wrap;
use crate::error::PyPolarsErr;
use crate::lazyframe::PyLazyFrame;

/// Registers the sqlglot renderer and the Python driver bridge. Idempotent.
pub fn register() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        register_sql_renderer("sqlglot", || {
            Ok(Arc::new(sqlglot::SqlglotRenderer::new()?) as _)
        });
        register_python_bridge(Arc::new(bridge::PyDriverBridge));
        // Like the ADBC driver manager: drivers installed into this Python environment
        // (`dbc install` in a venv) are found without `VIRTUAL_ENV` being set.
        let prefix = Python::attach(|py| -> PyResult<Option<String>> {
            let sys = py.import("sys")?;
            let prefix: String = sys.getattr("prefix")?.extract()?;
            let base: String = sys.getattr("base_prefix")?.extract()?;
            Ok((prefix != base).then_some(prefix))
        });
        if let Ok(Some(prefix)) = prefix {
            adbc::manifest::add_manifest_search_dir(
                std::path::Path::new(&prefix)
                    .join("etc")
                    .join("adbc")
                    .join("drivers"),
            );
        }
    });
}

fn value_err(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

#[pymethods]
impl PyLazyFrame {
    #[staticmethod]
    #[pyo3(signature = (
        source, source_type, connection, connection_kind, dialect, driver, connect_options,
        execute_options, python_options, credential_provider, params, schema, schema_overrides,
        partition_on, partition_auto, partition_num, partition_range, partition_predicates,
        order_by, consistency, max_connections, batch_size, pushdown,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new_from_database(
        source: &str,
        source_type: Option<&str>,
        connection: Py<PyAny>,
        connection_kind: &str,
        dialect: Option<&str>,
        driver: Option<String>,
        connect_options: Vec<(String, String)>,
        execute_options: Vec<(String, String)>,
        python_options: Option<Py<PyAny>>,
        credential_provider: Option<Py<PyAny>>,
        params: Option<Py<PyAny>>,
        schema: Option<Wrap<Schema>>,
        schema_overrides: Option<Wrap<Schema>>,
        partition_on: Option<String>,
        partition_auto: bool,
        partition_num: Option<usize>,
        partition_range: Option<(Wrap<AnyValue<'static>>, Wrap<AnyValue<'static>>)>,
        partition_predicates: Option<Vec<String>>,
        order_by: Vec<String>,
        consistency: &str,
        max_connections: Option<usize>,
        batch_size: Option<usize>,
        pushdown: bool,
        py: Python<'_>,
    ) -> PyResult<Self> {
        register();

        let target = match connection_kind {
            "uri" => ConnectionTarget::Uri(connection.extract::<String>(py)?.into()),
            "odbc" => ConnectionTarget::Odbc(connection.extract::<String>(py)?.into()),
            "factory" => ConnectionTarget::Factory(Arc::new(PythonObject(connection))),
            "live" => ConnectionTarget::Live(Arc::new(LiveConnection(PythonObject(connection)))),
            k => return Err(value_err(format!("invalid connection kind: {k}"))),
        };
        let connection = ConnectionSpec {
            target,
            connect_options: connect_options
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
            execute_options: execute_options
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
            credential_provider: credential_provider.map(|p| Arc::new(PythonObject(p))),
            query_params_key: params
                .as_ref()
                .map(|p| p.bind(py).repr().map(|r| r.to_string().into()))
                .transpose()?,
            query_params: params.map(|p| Arc::new(PythonObject(p))),
            python_options: python_options.map(|p| Arc::new(PythonObject(p))),
        };

        let source_type = source_type.filter(|s| *s != "auto");
        let relation = DatabaseRelation::parse(source, source_type);
        // A bad value of an argument is a ValueError, as for the other arguments.
        let dialect = dialect
            .map(DialectName::parse)
            .transpose()
            .map_err(value_err)?;

        // The Python side has checked the combinations.
        let partitioning = match (partition_predicates, partition_on.as_deref()) {
            (Some(preds), _) => {
                PartitionSpec::Predicates(preds.into_iter().map(Into::into).collect())
            },
            (None, None) => PartitionSpec::None,
            (None, Some(_)) if partition_auto => PartitionSpec::Auto {
                num_partitions: partition_num,
            },
            (None, Some(column)) => PartitionSpec::Column {
                column: column.into(),
                num_partitions: partition_num,
                range: partition_range.map(|(lo, hi)| {
                    let s = |v: Wrap<AnyValue<'static>>| {
                        let dtype = v.0.dtype();
                        Scalar::new(dtype, v.0)
                    };
                    Box::new((s(lo), s(hi)))
                }),
                strategy: PartitionStrategy::MinMax,
            },
        };

        let consistency = match consistency {
            "none" => Consistency::None,
            "snapshot" => Consistency::Snapshot,
            c => return Err(value_err(format!("invalid consistency: '{c}'"))),
        };

        let options = DatabaseScanOptions {
            connection,
            relation,
            dialect,
            driver: driver.map(Into::into),
            partitioning,
            order_by: order_by.into_iter().map(Into::into).collect(),
            consistency,
            fetch: FetchOptions {
                batch_size,
                max_connections,
            },
            pushdown,
            schema_overrides: schema_overrides.map(|s| Arc::new(s.0)),
        };

        let lf: LazyFrame = DslBuilder::scan_database(options, schema.map(|s| Arc::new(s.0)))
            .build()
            .into();
        Ok(lf.into())
    }
}

/// Rows of `show_drivers()`: `(dialect, backend, available, version, location, install)`.
#[pyfunction]
#[pyo3(signature = (dialect=None))]
#[allow(clippy::type_complexity)]
pub fn database_driver_statuses(
    py: Python<'_>,
    dialect: Option<&str>,
) -> PyResult<Vec<(String, String, bool, Option<String>, Option<String>, String)>> {
    register();
    let dialects = match dialect {
        Some(d) => vec![DialectName::parse(d).map_err(PyPolarsErr::from)?],
        None => DialectName::all().collect(),
    };
    py.detach(|| {
        Ok(dialects
            .into_iter()
            .flat_map(candidate_statuses)
            .map(|s| {
                (
                    s.dialect.to_string(),
                    s.kind.to_string(),
                    s.probe.is_some(),
                    s.probe
                        .as_ref()
                        .and_then(|p| p.version.as_ref().map(|v| v.to_string())),
                    s.probe
                        .as_ref()
                        .and_then(|p| p.location.as_ref().map(|v| v.to_string())),
                    install_hint(&s.kind),
                )
            })
            .collect())
    })
}

/// The dialect a URI scheme maps to.
#[pyfunction]
pub fn database_dialect_from_scheme(scheme: &str) -> String {
    DialectName::from_scheme(scheme).to_string()
}

/// Finds an installed ADBC driver (manifest search, then pip packages): `(library, entrypoint)`.
#[pyfunction]
pub fn database_find_adbc_driver(
    py: Python<'_>,
    name: &str,
) -> PyResult<Option<(String, Option<String>)>> {
    register();
    let name = name.to_string();
    py.detach(move || {
        if let Some(found) = adbc::find_driver(&name) {
            return Ok(Some((
                found.library.to_string_lossy().into_owned(),
                found.entrypoint,
            )));
        }
        let kind = BackendKind::Adbc {
            driver: name.into(),
        };
        Ok(probe(&kind)
            .and_then(|p| p.location)
            .map(|l| (l.to_string(), None)))
    })
}
