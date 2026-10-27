//! Database scans as a multi-scan: each source id is a partition, read by one
//! [`DatabasePartitionReader`].
//!
//! The SQL of every partition was rendered by the `compile_database_scans` optimizer pass. A
//! reader only opens a session, runs that SQL on a blocking thread, and casts the batches to
//! the planned schema.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use async_trait::async_trait;
use polars_async::executor::{JoinHandle, TaskMetricAggregator, TaskPriority};
use polars_async::{ASYNC, executor};
use polars_core::frame::DataFrame;
use polars_core::schema::SchemaRef;
use polars_error::{PolarsResult, polars_bail, polars_err};
use polars_io::cloud::CloudOptions;
use polars_io::database::backend::{
    CancelHandle, DatabaseBackend, DatabaseSession, RETRY_BACKOFF_MS, is_transient, open_backend,
    open_session_with_retries, retry_wait,
};
use polars_io::database::config::CompiledPartition;
use polars_io::database::schema::cast_to_schema;
use polars_io::database::{DatabaseScanConfig, catalog};
use polars_plan::dsl::ScanSource;
use polars_utils::IdxSize;
use polars_utils::index::idxsize_try_from;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::slice_enum::Slice;

use crate::morsel::{Morsel, MorselSeq, SourceToken, get_ideal_morsel_size};
use crate::nodes::io_sources::multi_scan::reader_interface::builder::FileReaderBuilder;
use crate::nodes::io_sources::multi_scan::reader_interface::capabilities::ReaderCapabilities;
use crate::nodes::io_sources::multi_scan::reader_interface::output::{
    FileReaderOutputRecv, FileReaderOutputSend,
};
use crate::nodes::io_sources::multi_scan::reader_interface::{
    BeginReadArgs, FileReader, FileReaderCallbacks, Projection,
};

/// Batches buffered between the fetch thread and the morsel task.
const FETCH_CHANNEL_CAPACITY: usize = 2;

/// Runs a driver call that may block (a cancel, closing a session) on a blocking thread,
/// never on the async executor's threads, without waiting for it.
fn run_detached(f: impl FnOnce() + Send + 'static) {
    drop(ASYNC.spawn_blocking(f));
}

/// A session leased from the scan's pool. Dropping it closes the session and frees its slot;
/// [`Shared::release`] keeps it open for the next partition instead.
struct Session {
    inner: Option<Box<dyn DatabaseSession>>,
    shared: Arc<Shared>,
}

impl std::ops::Deref for Session {
    type Target = dyn DatabaseSession;

    fn deref(&self) -> &Self::Target {
        self.inner.as_deref().unwrap()
    }
}

impl std::ops::DerefMut for Session {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.as_deref_mut().unwrap()
    }
}

impl Session {
    /// Closes the session and opens a new one in the same slot, so no other partition can
    /// take the slot in between.
    fn reconnect(&mut self) -> PolarsResult<()> {
        drop(self.inner.take());
        match self.shared.open_new() {
            Ok(s) => {
                self.inner = Some(s);
                Ok(())
            },
            Err(e) => {
                self.shared.free_slot();
                Err(e)
            },
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        if let Some(s) = self.inner.take() {
            drop(s);
            self.shared.free_slot();
        }
    }
}

/// Open sessions of one scan: never more than `max_connections` (idle ones included).
///
/// The multi-scan starts at most `max_connections` readers at once (the builder's
/// `max_concurrent_scans`, exact in ordered mode too), and each holds one session at a
/// time: a reader waits only for a session that is being given back, never for one held
/// by a reader whose output waits behind it.
#[derive(Default)]
struct Pool {
    idle: Vec<Box<dyn DatabaseSession>>,
    open: usize,
}

/// State shared by the readers of one scan.
struct Shared {
    config: Arc<DatabaseScanConfig>,
    /// Instantiated once per machine: re-checks that the planned backend exists here.
    backend: OnceLock<Arc<dyn DatabaseBackend>>,
    pool: Mutex<Pool>,
    pool_changed: std::sync::Condvar,
    /// Set when the scan is torn down: threads still waiting for a session give up instead
    /// of connecting for a query that is gone.
    aborted: AtomicBool,
}

impl Shared {
    fn open_new(&self) -> PolarsResult<Box<dyn DatabaseSession>> {
        let backend = self.backend()?;
        open_session_with_retries(
            backend.as_ref(),
            &self.config.options.connection,
            self.config.dialect,
        )
    }

    fn free_slot(&self) {
        self.pool.lock().unwrap().open -= 1;
        self.pool_changed.notify_all();
    }

    fn abort(&self) {
        self.aborted.store(true, Ordering::Relaxed);
        // under the lock, so a waiter can't miss the wakeup between its check and its wait
        let _pool = self.pool.lock().unwrap();
        self.pool_changed.notify_all();
    }

    fn backend(&self) -> PolarsResult<Arc<dyn DatabaseBackend>> {
        if let Some(b) = self.backend.get() {
            return Ok(b.clone());
        }
        let b = open_backend(&self.config.backend, self.config.options.fetch.batch_size)?;
        Ok(self.backend.get_or_init(|| b).clone())
    }

    /// An idle session, or a new one once fewer than `max_connections` are open. Blocks
    /// until then.
    fn acquire(self: &Arc<Self>) -> PolarsResult<Session> {
        let max = self.config.max_connections();
        let mut pool = self.pool.lock().unwrap();
        loop {
            if self.aborted.load(Ordering::Relaxed) {
                polars_bail!(ComputeError: "the database scan was stopped");
            }
            if let Some(s) = pool.idle.pop() {
                return Ok(Session {
                    inner: Some(s),
                    shared: self.clone(),
                });
            }
            if pool.open < max {
                pool.open += 1;
                drop(pool);
                return match self.open_new() {
                    Ok(s) => Ok(Session {
                        inner: Some(s),
                        shared: self.clone(),
                    }),
                    Err(e) => {
                        self.free_slot();
                        Err(e)
                    },
                };
            }
            pool = self.pool_changed.wait(pool).unwrap();
        }
    }

    /// Keeps a session open for the next partition.
    fn release(&self, mut session: Session) {
        if let Some(s) = session.inner.take() {
            self.pool.lock().unwrap().idle.push(s);
            self.pool_changed.notify_all();
        }
    }
}

pub struct DatabaseReaderBuilder {
    shared: Arc<Shared>,
    task_metrics: OnceLock<Arc<TaskMetricAggregator>>,
}

impl DatabaseReaderBuilder {
    pub fn new(config: Arc<DatabaseScanConfig>) -> Self {
        Self {
            shared: Arc::new(Shared {
                config,
                backend: OnceLock::new(),
                pool: Mutex::new(Pool::default()),
                pool_changed: std::sync::Condvar::new(),
                aborted: AtomicBool::new(false),
            }),
            task_metrics: OnceLock::new(),
        }
    }

    fn config(&self) -> &DatabaseScanConfig {
        &self.shared.config
    }
}

impl Drop for DatabaseReaderBuilder {
    fn drop(&mut self) {
        self.shared.abort();
    }
}

impl std::fmt::Debug for DatabaseReaderBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DatabaseReaderBuilder")
            .field("dialect", &self.config().dialect)
            .field("backend", &self.config().backend.kind)
            .field("partitions", &self.config().n_partitions())
            .finish()
    }
}

impl FileReaderBuilder for DatabaseReaderBuilder {
    fn reader_name(&self) -> PolarsResult<PlSmallStr> {
        Ok(format!(
            "database[{}/{}]",
            self.config().dialect,
            self.config().backend.kind
        )
        .into())
    }

    fn reader_capabilities(&self) -> PolarsResult<ReaderCapabilities> {
        use ReaderCapabilities as RC;
        let mut caps = RC::PRE_SLICE | RC::PARTIAL_FILTER | RC::UNORDERED_FILES;
        // The SQL owns the predicate only when every conjunct was pushed exactly.
        if self.config().compiled.as_ref().is_some_and(|c| c.all_exact) {
            caps |= RC::FULL_FILTER;
        }
        Ok(caps)
    }

    fn set_execution_state(&self, execution_state: &crate::execute::StreamingExecutionState) {
        if let Some(task_metrics) = execution_state.task_metrics.clone() {
            let _ = self.task_metrics.set(task_metrics);
        }
    }

    fn preferred_remote_pre_init(&self) -> Option<usize> {
        // Readers take a session only to count or read, so pre-initializing is free, but
        // more readers than sessions only queue.
        Some(self.config().max_connections())
    }

    fn sources_are_remote(&self) -> Option<bool> {
        Some(true)
    }

    fn max_concurrent_scans(&self) -> Option<usize> {
        Some(self.config().max_connections())
    }

    fn build_file_reader(
        &self,
        source: ScanSource,
        _cloud_options: Option<Arc<CloudOptions>>,
        _scan_source_idx: usize,
    ) -> PolarsResult<Box<dyn FileReader>> {
        let ScanSource::Path(path) = source else {
            polars_bail!(ComputeError: "database scan sources must be partition ids");
        };
        let id = path.as_str();
        let Some(compiled) = &self.config().compiled else {
            polars_bail!(
                ComputeError:
                "database scan was not compiled; the compile_database_scans optimizer pass did not run"
            );
        };
        // An absolute index into the partitions, never a position in the current sources.
        let partition = id
            .parse::<usize>()
            .ok()
            .and_then(|i| compiled.partitions.get(i).map(|p| (i, p.clone())))
            .ok_or_else(|| {
                polars_err!(
                    ComputeError:
                    "invalid database partition id '{id}' ({} partitions)",
                    compiled.partitions.len()
                )
            })?;
        Ok(Box::new(DatabasePartitionReader {
            shared: self.shared.clone(),
            index: partition.0,
            partition: Arc::new(partition.1),
            n_rows: None,
            task_metrics: self.task_metrics.get().cloned(),
        }))
    }
}

pub struct DatabasePartitionReader {
    shared: Arc<Shared>,
    index: usize,
    partition: Arc<CompiledPartition>,
    n_rows: Option<IdxSize>,
    task_metrics: Option<Arc<TaskMetricAggregator>>,
}

impl DatabasePartitionReader {
    fn context(&self) -> String {
        format!(
            "database partition {} via {}",
            self.index, self.shared.config.backend.kind
        )
    }

    async fn session(&self) -> PolarsResult<Session> {
        let shared = self.shared.clone();
        ASYNC
            .spawn_blocking(move || shared.acquire())
            .await
            .unwrap()
    }

    /// Runs the partition's `COUNT(*)`. Blocking, on a pool thread.
    async fn count(&mut self) -> PolarsResult<IdxSize> {
        if let Some(n) = self.n_rows {
            return Ok(n);
        }
        let mut session = self.session().await?;
        let sql = self.partition.count_sql.clone();
        let (session, result) = ASYNC
            .spawn_blocking(move || {
                let r = session.query(&sql).and_then(|df| catalog::scalar_u64(&df));
                // A failed session may be in any state: closed here, off the async threads.
                (r.is_ok().then_some(session), r)
            })
            .await
            .unwrap();
        // Back to the pool: holding it until this partition is read would let readers that
        // are counted out of order (a negative slice counts from the end) take every
        // session while the partition that is read next waits.
        if let Some(session) = session {
            self.shared.release(session);
        }
        let n = result.map_err(|e| e.context(self.context().into()))?;
        let n = idxsize_try_from(n).map_err(|_| {
            polars_err!(ComputeError: "database partition has {n} rows, which exceeds IdxSize; use bigidx")
        })?;
        self.n_rows = Some(n);
        Ok(n)
    }
}

/// Calls the current cancel handle on drop, unless disarmed. The cancel runs on a blocking
/// thread: drivers may open a connection to cancel (libpq) or take the GIL.
struct CancelOnDrop(Option<Arc<Mutex<Option<CancelHandle>>>>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        let Some(slot) = self.0.take() else { return };
        let Some(cancel) = slot.lock().unwrap().clone() else {
            return;
        };
        run_detached(move || {
            if let Err(e) = cancel()
                && polars_core::config::verbose()
            {
                eprintln!("[scan_database]: cancelling a statement failed: {e}");
            }
        });
    }
}

/// Positive-slice state over the stream of rows of one partition.
struct SliceState {
    skip: usize,
    remaining: usize,
}

impl SliceState {
    fn new(pre_slice: Option<Slice>) -> PolarsResult<Self> {
        Ok(match pre_slice {
            None => Self {
                skip: 0,
                remaining: usize::MAX,
            },
            Some(Slice::Positive { offset, len }) => Self {
                skip: offset,
                remaining: len,
            },
            Some(Slice::Negative { .. }) => {
                polars_bail!(ComputeError: "database reader was given a negative slice")
            },
        })
    }

    fn done(&self) -> bool {
        self.remaining == 0
    }

    fn apply(&mut self, df: DataFrame) -> DataFrame {
        let h = df.height();
        if self.skip >= h {
            self.skip -= h;
            return df.slice(0, 0);
        }
        let take = (h - self.skip).min(self.remaining);
        let out = df.slice(self.skip as i64, take);
        self.skip = 0;
        self.remaining -= take;
        out
    }
}

#[async_trait]
impl FileReader for DatabasePartitionReader {
    async fn initialize(&mut self) -> PolarsResult<()> {
        // Sessions are taken when counting or reading, never held by an idle reader.
        Ok(())
    }

    async fn file_schema(&mut self) -> PolarsResult<SchemaRef> {
        Ok(self.shared.config.schema.clone())
    }

    async fn n_rows_in_file(&mut self) -> PolarsResult<IdxSize> {
        self.count().await
    }

    fn begin_read(
        &mut self,
        args: BeginReadArgs,
    ) -> PolarsResult<(FileReaderOutputRecv, JoinHandle<PolarsResult<()>>)> {
        let BeginReadArgs {
            projection,
            row_index: _,
            pre_slice,
            predicate: _,
            callbacks:
                FileReaderCallbacks {
                    file_schema_tx,
                    n_rows_in_file_tx,
                    row_position_on_end_tx,
                },
            ..
        } = args;

        let projected = match projection {
            Projection::Plain(schema) => schema,
            Projection::Mapped { .. } => {
                polars_bail!(ComputeError: "database reader was given a mapped projection")
            },
        };
        let slice = SliceState::new(pre_slice)?;
        let (mut tx, rx) = FileReaderOutputSend::new_serial();

        let mut slf = DatabasePartitionReader {
            shared: self.shared.clone(),
            index: self.index,
            partition: self.partition.clone(),
            n_rows: self.n_rows,
            task_metrics: self.task_metrics.clone(),
        };
        let task_metrics = self.task_metrics.clone();

        let handle = executor::spawn(TaskPriority::Low, task_metrics.as_deref(), async move {
            let context = slf.context();
            let ctx = |e: polars_error::PolarsError| e.context(context.clone().into());

            if let Some(tx) = file_schema_tx {
                let _ = tx.send(slf.shared.config.schema.clone());
            }
            if let Some(tx) = n_rows_in_file_tx {
                let _ = tx.send(slf.count().await.map_err(ctx)?);
            }

            let source_token = SourceToken::new();
            let mut slice = slice;

            // Only the row count is needed: one `COUNT(*)` instead of reading rows.
            if projected.is_empty() {
                let n = slf.count().await.map_err(ctx)?;
                let df = slice.apply(DataFrame::empty_with_height(n as usize));
                if let Some(tx) = row_position_on_end_tx {
                    let _ = tx.send(n);
                }
                if df.height() > 0 {
                    let _ = tx
                        .send_morsel(Morsel::new_unregistered(
                            df,
                            MorselSeq::new(0),
                            source_token,
                        ))
                        .await;
                }
                return Ok(());
            }

            let session = slf.session().await.map_err(ctx)?;
            // The running statement's cancel handle; replaced when a retry reconnects.
            let cancel: Arc<Mutex<Option<CancelHandle>>> =
                Arc::new(Mutex::new(session.cancel_handle()));
            // Cancels the statement if this task is dropped mid-read (early stop of the whole
            // query), so the database doesn't keep working for nobody.
            let mut cancel_guard = CancelOnDrop(Some(cancel.clone()));
            let thread_cancel = cancel.clone();
            let sql = slf.partition.sql.clone();
            let batch_size = slf.shared.config.options.fetch.batch_size;
            // Reconnecting to a live connection gives back the same connection: no retries.
            let live = slf.shared.config.options.connection.is_live();
            let (batch_tx, mut batch_rx) =
                tokio::sync::mpsc::channel::<PolarsResult<DataFrame>>(FETCH_CHANNEL_CAPACITY);

            // Drivers block: the fetch loop runs on its own thread.
            let fetch = ASYNC.spawn_blocking(move || {
                let mut current = session;
                let mut attempt = 0;
                let retryable = |attempt: usize, e: &polars_error::PolarsError| {
                    !live && attempt < RETRY_BACKOFF_MS.len() && is_transient(e)
                };
                loop {
                    let mut sent = false;
                    let mut complete = false;
                    let mut retry_error = None;
                    match current.execute(&sql, batch_size) {
                        Ok(batches) => {
                            complete = true;
                            for batch in batches {
                                if let Err(e) = &batch
                                    && !sent
                                    && retryable(attempt, e)
                                {
                                    retry_error = Some(e.wrap_msg(|m| m.to_string()));
                                    complete = false;
                                    break;
                                }
                                let is_err = batch.is_err();
                                if batch_tx.blocking_send(batch).is_err() {
                                    // Nobody reads any more: stop the statement before
                                    // releasing it.
                                    if let Some(cancel) = &*thread_cancel.lock().unwrap() {
                                        let _ = cancel();
                                    }
                                    complete = false;
                                    break;
                                }
                                sent = true;
                                if is_err {
                                    complete = false;
                                    break;
                                }
                            }
                        },
                        Err(e) if retryable(attempt, &e) => retry_error = Some(e),
                        Err(e) => {
                            let _ = batch_tx.blocking_send(Err(e));
                        },
                    }
                    if let Some(e) = retry_error {
                        // Nothing was sent: retry on a new session, in the same pool slot. The
                        // old cancel handle holds the old connection open: drop it first.
                        *thread_cancel.lock().unwrap() = None;
                        retry_wait(attempt, &e);
                        attempt += 1;
                        // `reconnect` retries connecting itself.
                        if let Err(e) = current.reconnect() {
                            let _ = batch_tx.blocking_send(Err(e));
                            return None;
                        }
                        *thread_cancel.lock().unwrap() = current.cancel_handle();
                        continue;
                    }
                    return Some((current, complete));
                }
            });

            let morsel_size = get_ideal_morsel_size().max(1);
            let mut rows_read: u64 = 0;
            let mut seq: u64 = 0;
            let mut stopped = false;
            let mut error = None;

            'outer: while let Some(batch) = batch_rx.recv().await {
                let batch = match batch {
                    Ok(b) => b,
                    Err(e) => {
                        error = Some(e);
                        break;
                    },
                };
                rows_read += batch.height() as u64;
                let df = match cast_to_schema(&batch, &projected) {
                    Ok(df) => slice.apply(df),
                    Err(e) => {
                        error = Some(e);
                        break;
                    },
                };
                let mut offset = 0;
                while offset < df.height() {
                    let chunk = df.slice(offset as i64, morsel_size);
                    offset += chunk.height();
                    let morsel =
                        Morsel::new_unregistered(chunk, MorselSeq::new(seq), source_token.clone());
                    seq += 1;
                    if tx.send_morsel(morsel).await.is_err() || source_token.stop_requested() {
                        stopped = true;
                        break 'outer;
                    }
                }
                if slice.done() {
                    stopped = true;
                    break;
                }
            }

            if !(stopped || error.is_some()) {
                cancel_guard.0 = None;
            }
            // Cancels now, while the fetch thread may still be blocked in the driver.
            drop(cancel_guard);
            drop(batch_rx);
            let fetched = fetch.await.unwrap();
            let complete = fetched.as_ref().is_some_and(|(_, c)| *c) && !stopped && error.is_none();
            if let Some((session, _)) = fetched {
                if complete {
                    slf.shared.release(session);
                } else {
                    // A cancelled or failed session may be in any state: it is closed, not
                    // reused, and closing may block.
                    run_detached(move || drop(session));
                }
            }

            if let Some(e) = error {
                return Err(ctx(e));
            }
            // The count and the rows come from separate statements. Slices were planned from
            // the count: if the rows changed in between, they took the wrong rows.
            let limited = slf
                .shared
                .config
                .compiled
                .as_ref()
                .is_some_and(|c| c.limit.is_some());
            if complete
                && !limited
                && let Some(n) = slf.n_rows
                && rows_read != n as u64
            {
                return Err(ctx(polars_err!(
                    ComputeError:
                    "the partition returned {rows_read} rows, but COUNT(*) found {n}: the \
                    table changed during the scan"
                )));
            }
            if let Some(tx) = row_position_on_end_tx {
                let _ = tx.send(idxsize_try_from(rows_read).unwrap_or(IdxSize::MAX));
            }
            Ok(())
        });

        Ok((rx, handle))
    }
}
