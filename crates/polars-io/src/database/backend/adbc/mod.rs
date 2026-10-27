//! The ADBC backend: loads ADBC driver shared libraries through their C ABI and reads query
//! results as Arrow C streams.
//!
//! No ADBC driver manager is involved. Drivers are found through ADBC driver manifests (see
//! [`manifest`]) or given as a library path, `dlopen`ed, and initialised through their
//! `AdbcDriverInit`-style entrypoint. Drivers are never unloaded: Go drivers hang on unload.

mod ffi;
pub mod manifest;

use std::ffi::{CStr, CString, c_int, c_void};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::{Arc, LazyLock, Mutex};

use polars_arrow::array::{Array, StructArray};
use polars_arrow::datatypes::ArrowDataType;
use polars_arrow::ffi::{ArrowArrayStream, ArrowArrayStreamReader};
use polars_core::frame::DataFrame;
use polars_error::{PolarsResult, polars_bail, polars_err};

use self::ffi::*;
pub use self::manifest::{FoundDriver, find_driver, manifest_search_dirs};
use super::{
    BackendCapabilities, BackendChoice, BatchIter, CancelHandle, DatabaseBackend, DatabaseSession,
};
use crate::database::connection::{ConnectionSpec, ConnectionTarget};
use crate::database::dialect::DialectName;

/// An initialised driver. Never unloaded.
pub struct LoadedDriver {
    /// Boxed so its address is stable: handles point at it through `private_driver`.
    table: Box<AdbcDriver>,
    /// The ADBC API version the driver was initialised with.
    api_version: c_int,
    library: PathBuf,
    entrypoint: String,
}

// SAFETY: the function table is immutable after init, and ADBC drivers must be usable from any
// thread (handles are not shared between threads, except for cancel).
unsafe impl Send for LoadedDriver {}
unsafe impl Sync for LoadedDriver {}

impl LoadedDriver {
    pub fn library(&self) -> &Path {
        &self.library
    }

    pub fn entrypoint(&self) -> &str {
        &self.entrypoint
    }

    /// Whether the driver implements ADBC 1.1 (cancel, option getters, …).
    pub fn is_1_1(&self) -> bool {
        self.api_version >= ADBC_VERSION_1_1_0
    }

    fn table_ptr(&self) -> *mut AdbcDriver {
        &*self.table as *const AdbcDriver as *mut AdbcDriver
    }

    fn check(&self, status: AdbcStatusCode, error: &AdbcError, what: &str) -> PolarsResult<()> {
        if status == ADBC_STATUS_OK {
            return Ok(());
        }
        let message = error
            .describe()
            .unwrap_or_else(|| "no error message".into());
        Err(polars_err!(
            ComputeError: "ADBC {what} failed ({}) in '{}': {message}",
            status_name(status), self.library.display()
        ))
    }
}

macro_rules! driver_fn {
    ($driver:expr, $name:ident) => {
        $driver.table.$name.ok_or_else(|| {
            polars_err!(
                ComputeError: "ADBC driver '{}' does not implement {}",
                $driver.library.display(), stringify!($name)
            )
        })?
    };
}

/// Loaded drivers by (library, manifest entrypoint).
type DriverCache = polars_utils::aliases::PlHashMap<(PathBuf, Option<String>), Arc<LoadedDriver>>;

static LOADED_DRIVERS: LazyLock<Mutex<DriverCache>> = LazyLock::new(Default::default);

/// The init symbol derived from a library file name, as the ADBC driver manager does it:
/// `libadbc_driver_sqlite.so.1.0` → `AdbcDriverSqliteInit`, `proprietary_driver.dll` →
/// `AdbcProprietaryDriverInit`.
pub fn default_entrypoint(library: &str) -> String {
    let filename = library.rsplit(['/', '\\']).next().unwrap_or(library);
    let stem = filename.split('.').next().unwrap_or(filename);
    let stem = stem.strip_prefix("lib").unwrap_or(stem);
    let mut out: String = stem
        .split(['-', '_'])
        .filter(|t| !t.is_empty())
        .map(|t| {
            let mut c = t.chars();
            let first = c.next().unwrap().to_ascii_uppercase();
            std::iter::once(first).chain(c).collect::<String>()
        })
        .collect();
    if !out.starts_with("Adbc") {
        out.insert_str(0, "Adbc");
    }
    out.push_str("Init");
    out
}

/// Loads and initialises a driver library, or returns the one already loaded.
pub fn load_driver(found: &FoundDriver) -> PolarsResult<Arc<LoadedDriver>> {
    let key_path = found
        .library
        .canonicalize()
        .unwrap_or_else(|_| found.library.clone());
    let key = (key_path, found.entrypoint.clone());
    let mut cache = LOADED_DRIVERS.lock().unwrap();
    if let Some(d) = cache.get(&key) {
        return Ok(d.clone());
    }
    let driver = Arc::new(load_driver_uncached(found)?);
    cache.insert(key, driver.clone());
    Ok(driver)
}

fn load_driver_uncached(found: &FoundDriver) -> PolarsResult<LoadedDriver> {
    let path = &found.library;
    // SAFETY: loading a library runs its initialisers; the user asked for this driver.
    let lib = unsafe { libloading::Library::new(path.as_path()) }.map_err(|e| {
        polars_err!(ComputeError: "could not load ADBC driver library '{}': {e}", path.display())
    })?;

    let mut candidates = vec![];
    candidates.extend(found.entrypoint.clone());
    candidates.push(default_entrypoint(&path.to_string_lossy()));
    candidates.push("AdbcDriverInit".to_string());
    candidates.dedup();

    let mut init: Option<(AdbcDriverInitFunc, String)> = None;
    for name in &candidates {
        // SAFETY: the symbol is an `AdbcDriverInitFunc` by the ADBC convention.
        if let Ok(sym) = unsafe { lib.get::<AdbcDriverInitFunc>(name.as_str()) } {
            init = Some((*sym, name.clone()));
            break;
        }
    }
    let Some((init, entrypoint)) = init else {
        polars_bail!(
            ComputeError: "ADBC driver library '{}' exports none of the entrypoints {:?}",
            path.display(), candidates
        )
    };

    let mut api_version = ADBC_VERSION_1_1_0;
    let mut table = Box::new(AdbcDriver::empty());
    let mut error = AdbcError::new();
    let mut status = unsafe {
        init(
            api_version,
            &mut *table as *mut AdbcDriver as *mut c_void,
            &mut error,
        )
    };
    if status == ADBC_STATUS_NOT_IMPLEMENTED {
        api_version = ADBC_VERSION_1_0_0;
        *table = AdbcDriver::empty();
        error.reset();
        status = unsafe {
            init(
                api_version,
                &mut *table as *mut AdbcDriver as *mut c_void,
                &mut error,
            )
        };
    }
    if status != ADBC_STATUS_OK {
        let message = error.describe().unwrap_or_default();
        polars_bail!(
            ComputeError: "ADBC driver '{}' failed to initialise ({}) via '{entrypoint}': {message}",
            path.display(), status_name(status)
        );
    }
    drop(error);

    // Never unloaded: Go drivers hang on unload, and other threads may still run driver code.
    std::mem::forget(lib);

    Ok(LoadedDriver {
        table,
        api_version,
        library: path.clone(),
        entrypoint,
    })
}

pub struct AdbcBackend {
    driver: Arc<LoadedDriver>,
    found: FoundDriver,
    choice: BackendChoice,
}

impl AdbcBackend {
    /// `driver`: an ADBC driver name (`"sqlite"`, `"postgresql"`, …), or a path to a shared
    /// library or a `.toml` manifest.
    /// `location_hint`: a library or manifest path found during planning, possibly on another
    /// machine. Tried first when it exists here.
    pub fn load(
        driver: &str,
        location_hint: Option<&str>,
        choice: BackendChoice,
    ) -> PolarsResult<Self> {
        let found = location_hint
            .and_then(find_driver)
            .or_else(|| find_driver(driver));
        let Some(found) = found else {
            let dirs = manifest_search_dirs()
                .iter()
                .map(|d| d.display().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            let hint = location_hint
                .map(|h| format!("'{h}' (from planning) does not exist here; "))
                .unwrap_or_default();
            if driver.contains(['/', '\\']) || driver.ends_with(".toml") {
                polars_bail!(ComputeError: "ADBC driver library or manifest '{driver}' does not exist");
            }
            polars_bail!(
                ComputeError:
                "ADBC driver '{driver}' not found: {hint}looked for '{driver}.toml' in [{dirs}]. \
                Install it with: {}  (or pass the path of the driver library or manifest)",
                super::resolve::adbc_install_hint(driver)
            )
        };
        let driver = load_driver(&found)?;
        Ok(Self {
            driver,
            found,
            choice,
        })
    }

    pub fn loaded_driver(&self) -> &Arc<LoadedDriver> {
        &self.driver
    }

    pub fn found(&self) -> &FoundDriver {
        &self.found
    }

    /// Opens a connection. Like [`DatabaseBackend::connect`], with the concrete session type.
    pub fn open(&self, spec: &ConnectionSpec, dialect: DialectName) -> PolarsResult<AdbcSession> {
        let mut uri = adbc_uri(spec, dialect)?;
        if matches!(dialect, DialectName::Postgresql | DialectName::Redshift)
            && let Some(u) = &mut uri
            && !u.contains("connect_timeout=")
        {
            // libpq otherwise waits for the OS TCP timeout (minutes), and connects are retried.
            u.push(if u.contains('?') { '&' } else { '?' });
            u.push_str("connect_timeout=10");
        }
        let mut options: Vec<(CString, CString)> = vec![];
        let mut creds = credentials(spec)?;
        if dialect == DialectName::Clickhouse
            && let Some(u) = &uri
            && let Some(http) = clickhouse_http_uri(u)
        {
            // The ClickHouse driver takes the HTTP endpoint, with credentials as options.
            uri = Some(http.uri);
            let (user, password) = creds.take().unwrap_or_default();
            for (key, value) in [
                (ADBC_OPTION_USERNAME, user.or(http.user)),
                (ADBC_OPTION_PASSWORD, password.or(http.password)),
            ] {
                if let Some(v) = value {
                    options.push((key.to_owned(), cstring(&v, "credential")?));
                }
            }
        }
        if let Some((user, password)) = creds {
            // Drivers that take a `scheme://` URI (libpq, MySQL, …) read credentials from it
            // and may reject the `username`/`password` options.
            match uri
                .as_deref()
                .and_then(|u| with_userinfo(u, user.as_deref(), password.as_deref()))
            {
                Some(with_creds) => uri = Some(with_creds),
                None => {
                    if let Some(u) = user {
                        options.push((ADBC_OPTION_USERNAME.to_owned(), cstring(&u, "credential")?));
                    }
                    if let Some(p) = password {
                        options.push((ADBC_OPTION_PASSWORD.to_owned(), cstring(&p, "credential")?));
                    }
                },
            }
        }
        if let Some(uri) = uri {
            options.insert(0, (ADBC_OPTION_URI.to_owned(), cstring(&uri, "uri")?));
        }
        for (k, v) in &spec.connect_options {
            options.push((cstring(k, "option key")?, cstring(v, k)?));
        }
        let conn = Conn::open(self.driver.clone(), &options)?;
        let execute_options = spec
            .execute_options
            .iter()
            .map(|(k, v)| Ok((cstring(k, "option key")?, cstring(v, k)?)))
            .collect::<PolarsResult<Vec<_>>>()?;
        Ok(AdbcSession {
            conn: Arc::new(conn),
            dialect,
            execute_options,
        })
    }
}

impl DatabaseBackend for AdbcBackend {
    fn choice(&self) -> &BackendChoice {
        &self.choice
    }

    fn capabilities(&self) -> BackendCapabilities {
        let mut caps = BackendCapabilities::ARROW_NATIVE
            | BackendCapabilities::STREAMING
            | BackendCapabilities::NO_GIL
            | BackendCapabilities::SESSION_SETUP;
        if self.driver.is_1_1() {
            caps |= BackendCapabilities::CANCEL;
        }
        caps
    }

    fn connect(
        &self,
        spec: &ConnectionSpec,
        dialect: DialectName,
    ) -> PolarsResult<Box<dyn DatabaseSession>> {
        Ok(Box::new(self.open(spec, dialect)?))
    }
}

fn cstring(s: &str, what: &str) -> PolarsResult<CString> {
    CString::new(s).map_err(|_| polars_err!(ComputeError: "ADBC {what} contains a NUL byte"))
}

/// The `uri` database option for a connection target.
fn adbc_uri(spec: &ConnectionSpec, dialect: DialectName) -> PolarsResult<Option<String>> {
    match &spec.target {
        ConnectionTarget::Uri(_) => {
            let uri = spec.uri_without_scheme_driver().unwrap_or_default();
            if dialect == DialectName::Sqlite {
                return Ok(Some(sqlite_path(&uri)));
            }
            Ok(Some(uri))
        },
        ConnectionTarget::Odbc(_) => polars_bail!(
            InvalidOperation: "an ODBC connection string can't be used with an ADBC driver"
        ),
        #[allow(unreachable_patterns)]
        _ => polars_bail!(
            InvalidOperation: "a Python connection object can't be used with an ADBC driver"
        ),
    }
}

struct ClickhouseHttp {
    uri: String,
    user: Option<String>,
    password: Option<String>,
}

/// `clickhouse://user:pass@host:8123/db` → `http://host:8123/?database=db` and the
/// credentials; `clickhouses://` (or `?secure=true`) → `https://`. `http(s)://` passes through.
fn clickhouse_http_uri(uri: &str) -> Option<ClickhouseHttp> {
    let (scheme, rest) = uri.split_once("://")?;
    let scheme = scheme.to_ascii_lowercase();
    if scheme == "http" || scheme == "https" {
        return None;
    }
    let (rest, query) = rest.split_once('?').unwrap_or((rest, ""));
    let (authority, path) = rest.split_once('/').unwrap_or((rest, ""));
    let (userinfo, host) = match authority.rsplit_once('@') {
        Some((u, h)) => (Some(u), h),
        None => (None, authority),
    };
    let (user, password) = match userinfo {
        Some(u) => match u.split_once(':') {
            Some((u, p)) => (Some(percent_decode(u)), Some(percent_decode(p))),
            None => (Some(percent_decode(u)), None),
        },
        None => (None, None),
    };
    let secure = scheme.ends_with('s')
        || query
            .split('&')
            .any(|kv| kv.eq_ignore_ascii_case("secure=true") || kv == "secure=1");
    let mut params: Vec<String> = query
        .split('&')
        .filter(|kv| !kv.is_empty() && !kv.to_ascii_lowercase().starts_with("secure="))
        .map(str::to_string)
        .collect();
    let database = path.trim_matches('/');
    if !database.is_empty() {
        params.insert(0, format!("database={database}"));
    }
    let mut out = format!("{}://{host}", if secure { "https" } else { "http" });
    if !params.is_empty() {
        out.push_str("/?");
        out.push_str(&params.join("&"));
    }
    Some(ClickhouseHttp {
        uri: out,
        user,
        password,
    })
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let Ok(b) = u8::from_str_radix(&s[i + 1..i + 3], 16)
        {
            out.push(b);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// `sqlite:///path/to.db` → `/path/to.db`; `sqlite://` → `:memory:`. `file:` URIs and plain
/// paths pass through.
fn sqlite_path(uri: &str) -> String {
    let lower = uri.to_ascii_lowercase();
    let rest = ["sqlite3://", "sqlite://", "sqlite3:", "sqlite:"]
        .iter()
        .find(|p| lower.starts_with(*p))
        .map(|p| &uri[p.len()..]);
    match rest {
        None => uri.to_string(),
        Some("" | "/" | ":memory:" | "/:memory:") => ":memory:".to_string(),
        Some(path) => path.to_string(),
    }
}

/// `username` / `password` database options from the connection's credential provider.
fn credentials(spec: &ConnectionSpec) -> PolarsResult<Option<(Option<String>, Option<String>)>> {
    #[cfg(feature = "python")]
    if spec.credential_provider.is_some() {
        let Some(bridge) = super::python_bridge() else {
            polars_bail!(ComputeError: "a credential provider needs Python, which is not available");
        };
        return bridge.credentials(spec);
    }
    let _ = spec;
    Ok(None)
}

/// `scheme://user:password@host/…` with the given credentials, replacing those in the URI.
/// `None` if the URI has no authority part.
fn with_userinfo(uri: &str, user: Option<&str>, password: Option<&str>) -> Option<String> {
    let (scheme, rest) = uri.split_once("://")?;
    let auth_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let (authority, tail) = rest.split_at(auth_end);
    let (old_userinfo, host) = match authority.rsplit_once('@') {
        Some((u, h)) => (Some(u), h),
        None => (None, authority),
    };
    let old_user = old_userinfo.map(|u| u.split_once(':').map_or(u, |(u, _)| u));
    let user = match user {
        Some(u) => Some(percent_encode(u)),
        None => old_user.map(str::to_string),
    };
    let mut userinfo = user.unwrap_or_default();
    if let Some(p) = password {
        userinfo.push(':');
        userinfo.push_str(&percent_encode(p));
    }
    let at = if userinfo.is_empty() { "" } else { "@" };
    Some(format!("{scheme}://{userinfo}{at}{host}{tail}"))
}

fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// A database and its one connection. Shared by the session, its result streams and its
/// cancel handle, so it lives until all of them are gone.
struct Conn {
    driver: Arc<LoadedDriver>,
    database: *mut AdbcDatabase,
    connection: *mut AdbcConnection,
    /// The statement currently executing, for cancel.
    active: Mutex<Option<StmtPtr>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct StmtPtr(*mut AdbcStatement);
unsafe impl Send for StmtPtr {}

// SAFETY: the connection is used by one thread at a time (the session is `&mut`, and result
// streams are consumed in order); ADBC allows cancel from another thread.
unsafe impl Send for Conn {}
unsafe impl Sync for Conn {}

impl Conn {
    fn open(driver: Arc<LoadedDriver>, options: &[(CString, CString)]) -> PolarsResult<Self> {
        let d = &*driver;
        let mut error = AdbcError::new();

        let database = Box::into_raw(Box::new(AdbcDatabase::empty()));
        let connection = Box::into_raw(Box::new(AdbcConnection::empty()));
        // From here on, `Drop` cleans up whatever was initialised.
        let conn = Conn {
            driver: driver.clone(),
            database,
            connection,
            active: Mutex::new(None),
        };

        unsafe {
            let status = driver_fn!(d, DatabaseNew)(database, &mut error);
            d.check(status, &error, "DatabaseNew")?;
            (*database).private_driver = d.table_ptr();
            let set = driver_fn!(d, DatabaseSetOption);
            for (k, v) in options {
                error.reset();
                let status = set(database, k.as_ptr(), v.as_ptr(), &mut error);
                d.check(
                    status,
                    &error,
                    &format!("setting database option '{}'", k.to_string_lossy()),
                )?;
            }
            error.reset();
            let status = driver_fn!(d, DatabaseInit)(database, &mut error);
            d.check(status, &error, "DatabaseInit")?;

            error.reset();
            let status = driver_fn!(d, ConnectionNew)(connection, &mut error);
            d.check(status, &error, "ConnectionNew")?;
            (*connection).private_driver = d.table_ptr();
            error.reset();
            let status = driver_fn!(d, ConnectionInit)(connection, database, &mut error);
            d.check(status, &error, "ConnectionInit")?;
        }
        Ok(conn)
    }

    fn cancel(&self) -> PolarsResult<()> {
        let d = &*self.driver;
        let active = self.active.lock().unwrap();
        let mut error = AdbcError::new();
        let mut status = ADBC_STATUS_NOT_IMPLEMENTED;
        unsafe {
            if let (Some(stmt), Some(cancel)) = (*active, d.table.StatementCancel) {
                status = cancel(stmt.0, &mut error);
            }
            if status == ADBC_STATUS_NOT_IMPLEMENTED
                && let Some(cancel) = d.table.ConnectionCancel
            {
                error.reset();
                status = cancel(self.connection, &mut error);
            }
        }
        // Nothing running: nothing to cancel.
        if status == ADBC_STATUS_INVALID_STATE {
            return Ok(());
        }
        d.check(status, &error, "cancel")
    }
}

impl Drop for Conn {
    fn drop(&mut self) {
        let t = &self.driver.table;
        let mut error = AdbcError::new();
        unsafe {
            if !(*self.connection).private_data.is_null()
                && let Some(release) = t.ConnectionRelease
            {
                release(self.connection, &mut error);
                error.reset();
            }
            if !(*self.database).private_data.is_null()
                && let Some(release) = t.DatabaseRelease
            {
                release(self.database, &mut error);
            }
            drop(Box::from_raw(self.connection));
            drop(Box::from_raw(self.database));
        }
    }
}

/// An open statement. Released on drop.
struct Stmt {
    conn: Arc<Conn>,
    ptr: *mut AdbcStatement,
}

unsafe impl Send for Stmt {}

impl Stmt {
    fn new(conn: Arc<Conn>) -> PolarsResult<Self> {
        let d = &*conn.driver;
        let ptr = Box::into_raw(Box::new(AdbcStatement::empty()));
        let stmt = Stmt {
            conn: conn.clone(),
            ptr,
        };
        let mut error = AdbcError::new();
        unsafe {
            let status = driver_fn!(d, StatementNew)(conn.connection, ptr, &mut error);
            d.check(status, &error, "StatementNew")?;
            (*ptr).private_driver = d.table_ptr();
        }
        Ok(stmt)
    }

    fn driver(&self) -> &LoadedDriver {
        &self.conn.driver
    }

    fn set_option(&self, key: &CStr, value: &CStr) -> PolarsResult<()> {
        let d = self.driver();
        let mut error = AdbcError::new();
        let status = unsafe {
            driver_fn!(d, StatementSetOption)(self.ptr, key.as_ptr(), value.as_ptr(), &mut error)
        };
        d.check(
            status,
            &error,
            &format!("setting statement option '{}'", key.to_string_lossy()),
        )
    }

    fn set_sql(&self, sql: &str) -> PolarsResult<()> {
        let d = self.driver();
        let sql = cstring(sql, "query")?;
        let mut error = AdbcError::new();
        let status =
            unsafe { driver_fn!(d, StatementSetSqlQuery)(self.ptr, sql.as_ptr(), &mut error) };
        d.check(status, &error, "StatementSetSqlQuery")
    }

    /// Runs the statement. With `out` null, no result set is expected.
    fn execute(&self, out: *mut ArrowArrayStream) -> PolarsResult<()> {
        let d = self.driver();
        let execute = driver_fn!(d, StatementExecuteQuery);
        *self.conn.active.lock().unwrap() = Some(StmtPtr(self.ptr));
        let mut error = AdbcError::new();
        let mut rows_affected: i64 = -1;
        let status = unsafe { execute(self.ptr, out, &mut rows_affected, &mut error) };
        if out.is_null() {
            *self.conn.active.lock().unwrap() = None;
        }
        d.check(status, &error, "query")
    }
}

impl Drop for Stmt {
    fn drop(&mut self) {
        {
            let mut active = self.conn.active.lock().unwrap();
            if *active == Some(StmtPtr(self.ptr)) {
                *active = None;
            }
        }
        unsafe {
            if !(*self.ptr).private_data.is_null()
                && let Some(release) = self.conn.driver.table.StatementRelease
            {
                let mut error = AdbcError::new();
                release(self.ptr, &mut error);
            }
            drop(Box::from_raw(self.ptr));
        }
    }
}

/// A connection to a database through an ADBC driver.
pub struct AdbcSession {
    conn: Arc<Conn>,
    dialect: DialectName,
    execute_options: Vec<(CString, CString)>,
}

impl AdbcSession {
    fn prepare(&self, sql: &str, batch_size: Option<usize>) -> PolarsResult<Stmt> {
        let stmt = Stmt::new(self.conn.clone())?;
        if let Some(n) = batch_size
            && let Some((key, value)) = batch_size_option(self.dialect, n)
        {
            // A hint: drivers that reject it still work.
            let _ = stmt.set_option(key, &CString::new(value).unwrap());
        }
        for (k, v) in &self.execute_options {
            stmt.set_option(k, v)?;
        }
        stmt.set_sql(sql)?;
        Ok(stmt)
    }
}

/// The statement option that bounds the batch size, for drivers known to have one.
fn batch_size_option(dialect: DialectName, rows: usize) -> Option<(&'static CStr, String)> {
    match dialect {
        DialectName::Sqlite => Some((c"adbc.sqlite.query.batch_rows", rows.to_string())),
        // A byte hint: assume narrow rows, so a batch never has far more rows than asked.
        DialectName::Postgresql | DialectName::Redshift => Some((
            c"adbc.postgresql.batch_size_hint_bytes",
            rows.saturating_mul(64).max(1024).to_string(),
        )),
        _ => None,
    }
}

impl DatabaseSession for AdbcSession {
    fn execute(&mut self, sql: &str, batch_size: Option<usize>) -> PolarsResult<BatchIter> {
        let stmt = self.prepare(sql, batch_size)?;
        let stream = Box::into_raw(Box::new(ArrowArrayStream::empty()));
        let result = stmt.execute(stream);
        // SAFETY: allocated above; the driver only wrote into it.
        let stream = unsafe { Box::from_raw(stream) };
        result?;
        // The heap address stays valid while the reader owns the box.
        let stream_ptr = &*stream as *const ArrowArrayStream as *mut ArrowArrayStream;
        // SAFETY: the driver returned a valid C stream.
        let reader = unsafe { ArrowArrayStreamReader::try_new(stream) }?;
        // Before any array is imported or built: those of these types would panic.
        check_arrow_types(&reader.field().dtype)?;
        Ok(Box::new(AdbcBatches {
            reader: Some(reader),
            stream_ptr,
            stmt,
            yielded: false,
        }))
    }

    fn execute_update(&mut self, sql: &str) -> PolarsResult<()> {
        let stmt = self.prepare(sql, None)?;
        stmt.execute(ptr::null_mut())
    }

    fn cancel_handle(&self) -> Option<CancelHandle> {
        if !self.conn.driver.is_1_1() {
            return None;
        }
        let conn = self.conn.clone();
        Some(Arc::new(move || conn.cancel()))
    }
}

/// The driver's error for a failed stream operation (ADBC 1.1), which carries more than the
/// stream's own `get_last_error`.
fn stream_error(stmt: &Stmt, stream: *mut ArrowArrayStream) -> Option<polars_error::PolarsError> {
    let d = stmt.driver();
    let f = d.table.ErrorFromArrayStream?;
    let mut status = ADBC_STATUS_OK;
    let error = unsafe { f(stream, &mut status) };
    if error.is_null() {
        return None;
    }
    // Owned by the stream: borrowed, never released here.
    let message = unsafe { &*error }.describe()?;
    Some(polars_err!(
        ComputeError: "ADBC query failed ({}) in '{}': {message}",
        status_name(status), d.library.display()
    ))
}

/// The batches of one query, as `DataFrame`s.
struct AdbcBatches {
    // Field order is drop order: the stream is released before its statement.
    reader: Option<ArrowArrayStreamReader<Box<ArrowArrayStream>>>,
    stream_ptr: *mut ArrowArrayStream,
    stmt: Stmt,
    yielded: bool,
}

// SAFETY: the stream and statement are only used from the thread that pulls the iterator.
unsafe impl Send for AdbcBatches {}

impl Iterator for AdbcBatches {
    type Item = PolarsResult<DataFrame>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let reader = self.reader.as_mut()?;
            // SAFETY: the driver produces arrays matching its schema.
            match unsafe { reader.next() } {
                Some(Ok(array)) => {
                    if array.is_empty() {
                        continue;
                    }
                    self.yielded = true;
                    return Some(batch_to_df(array));
                },
                Some(Err(e)) => {
                    let err = stream_error(&self.stmt, self.stream_ptr).unwrap_or(e);
                    self.reader = None;
                    return Some(Err(err));
                },
                None => {
                    let reader = self.reader.take()?;
                    if self.yielded {
                        return None;
                    }
                    // Empty result: one empty frame with the result's columns.
                    self.yielded = true;
                    let dtype = reader.field().dtype.clone();
                    return Some(batch_to_df(Box::new(StructArray::new_empty(dtype))));
                },
            }
        }
    }
}

/// Arrow types Polars has no dtype for (importing them would panic).
fn unsupported_arrow_type(dtype: &ArrowDataType) -> bool {
    match dtype {
        ArrowDataType::Interval(_) | ArrowDataType::Union(_) | ArrowDataType::Decimal256(_, _) => {
            true
        },
        ArrowDataType::List(f)
        | ArrowDataType::LargeList(f)
        | ArrowDataType::FixedSizeList(f, _)
        | ArrowDataType::Map(f, _) => unsupported_arrow_type(f.dtype()),
        ArrowDataType::Struct(fields) => fields.iter().any(|f| unsupported_arrow_type(f.dtype())),
        ArrowDataType::Extension(ext) => unsupported_arrow_type(&ext.inner),
        _ => false,
    }
}

/// An error for a result column of a type Polars has no dtype for.
fn check_arrow_types(dtype: &ArrowDataType) -> PolarsResult<()> {
    let ArrowDataType::Struct(fields) = dtype else {
        polars_bail!(ComputeError: "ADBC driver returned a non-struct batch: {dtype:?}");
    };
    if let Some(f) = fields.iter().find(|f| unsupported_arrow_type(f.dtype())) {
        polars_bail!(
            ComputeError:
            "column '{}' has Arrow type {:?}, which Polars can't read; convert it in a query, \
            e.g. CAST(\"{}\" AS TEXT)",
            f.name, f.dtype(), f.name
        );
    }
    Ok(())
}

fn batch_to_df(array: Box<dyn Array>) -> PolarsResult<DataFrame> {
    check_arrow_types(array.dtype())?;
    // A record batch has no top-level nulls; drop a validity some drivers still attach.
    let array = array.with_validity(None);
    let array = array
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("struct dtype")
        .clone();
    DataFrame::try_from(array)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entrypoints() {
        assert_eq!(
            default_entrypoint("/x/libadbc_driver_sqlite.so"),
            "AdbcDriverSqliteInit"
        );
        assert_eq!(
            default_entrypoint("libadbc_driver_sqlite.so.2.0.0"),
            "AdbcDriverSqliteInit"
        );
        assert_eq!(
            default_entrypoint("C:\\d\\adbc_driver_sqlite.dll"),
            "AdbcDriverSqliteInit"
        );
        assert_eq!(
            default_entrypoint("proprietary_driver.dll"),
            "AdbcProprietaryDriverInit"
        );
        assert_eq!(
            default_entrypoint("libadbc-driver-flight-sql.dylib"),
            "AdbcDriverFlightSqlInit"
        );
    }

    /// Reads through ADBC with no Python in the process: the Rust-only path workers use.
    /// Set `POLARS_TEST_ADBC_SQLITE_LIBRARY` (and for the servers
    /// `POLARS_TEST_ADBC_POSTGRES_LIBRARY` + `POLARS_TEST_POSTGRES_URI`,
    /// `POLARS_TEST_ADBC_MYSQL_LIBRARY` + `POLARS_TEST_MYSQL_URI`) to run it.
    #[test]
    fn reads_without_python() {
        use polars_core::prelude::DataType;

        use crate::database::backend::{BackendChoice, BackendKind};

        let read = |lib_var: &str, uri: &str, dialect: DialectName, sql: &str| {
            let Ok(lib) = std::env::var(lib_var) else {
                return None;
            };
            let choice = BackendChoice {
                kind: BackendKind::Adbc {
                    driver: lib.as_str().into(),
                },
                version: None,
                location: None,
                reason: "test".into(),
            };
            let backend = AdbcBackend::load(&lib, None, choice).unwrap();
            let spec = ConnectionSpec::new(ConnectionTarget::Uri(uri.into()));
            let mut session = backend.open(&spec, dialect).unwrap();
            Some(session.query(sql).unwrap())
        };

        if let Some(df) = read(
            "POLARS_TEST_ADBC_SQLITE_LIBRARY",
            "sqlite://",
            DialectName::Sqlite,
            "SELECT 1 AS a, 'x' AS b, 1.5 AS c, NULL AS d",
        ) {
            assert_eq!(df.shape(), (1, 4));
            assert_eq!(df.column("a").unwrap().dtype(), &DataType::Int64);
            assert_eq!(df.column("b").unwrap().dtype(), &DataType::String);
        }
        // Every column of `edge_types` (JSON, UUID, binary, NUMERIC, tz, time) imports.
        for (lib, uri, dialect) in [
            (
                "POLARS_TEST_ADBC_POSTGRES_LIBRARY",
                "POLARS_TEST_POSTGRES_URI",
                DialectName::Postgresql,
            ),
            (
                "POLARS_TEST_ADBC_MYSQL_LIBRARY",
                "POLARS_TEST_MYSQL_URI",
                DialectName::Mysql,
            ),
        ] {
            let Ok(uri) = std::env::var(uri) else {
                continue;
            };
            let edge = dialect.quote_ident("edge_types");
            if let Some(df) = read(lib, &uri, dialect, &format!("SELECT * FROM {edge}")) {
                assert_eq!(df.shape(), (3, 7), "{dialect}");
                for c in df.columns().iter().filter(|c| c.name() != "id") {
                    assert!(
                        c.null_count() >= 1,
                        "{dialect}: {} has no NULL row",
                        c.name()
                    );
                }
            }
        }
    }

    #[test]
    fn clickhouse_uris() {
        let h = clickhouse_http_uri("clickhouse://u:p%40ss@h:8123/db").unwrap();
        assert_eq!(h.uri, "http://h:8123/?database=db");
        assert_eq!(h.user.as_deref(), Some("u"));
        assert_eq!(h.password.as_deref(), Some("p@ss"));
        let h = clickhouse_http_uri("clickhouse://h:8443?secure=true").unwrap();
        assert_eq!(h.uri, "https://h:8443");
        assert!(clickhouse_http_uri("http://h:8123").is_none());
    }

    #[test]
    fn userinfo() {
        assert_eq!(
            with_userinfo("postgresql://u@h:5432/db", None, Some("p@ss:w/rd")).as_deref(),
            Some("postgresql://u:p%40ss%3Aw%2Frd@h:5432/db")
        );
        assert_eq!(
            with_userinfo("postgresql://old:x@h/db?a=1", Some("new"), Some("y")).as_deref(),
            Some("postgresql://new:y@h/db?a=1")
        );
        assert_eq!(
            with_userinfo("mysql://h", Some("u"), None).as_deref(),
            Some("mysql://u@h")
        );
        assert_eq!(with_userinfo("/tmp/a.db", Some("u"), None), None);
    }

    #[test]
    fn sqlite_paths() {
        assert_eq!(sqlite_path("sqlite:///tmp/a.db"), "/tmp/a.db");
        assert_eq!(sqlite_path("sqlite://"), ":memory:");
        assert_eq!(sqlite_path("sqlite:///:memory:"), ":memory:");
        assert_eq!(sqlite_path("file:a.db?mode=ro"), "file:a.db?mode=ro");
    }
}
