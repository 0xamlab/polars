"""The `scan_database` arguments: their checks, and how each connection form is used."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import date
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("sqlglot")


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute('CREATE TABLE t (id INTEGER PRIMARY KEY, s TEXT, "auto" INTEGER)')
        conn.executemany(
            "INSERT INTO t VALUES (?, ?, ?)", [(i, f"v{i}", i) for i in range(1, 101)]
        )
        conn.commit()
    return path


@pytest.fixture
def uri(db: Path) -> str:
    return f"sqlite:///{db}"


@pytest.mark.parametrize(
    "connection",
    [
        "polars:s3cret@127.0.0.1:5432/polars",  # no scheme
        "postgresql:polars:s3cret@127.0.0.1/polars",  # no `//`
        "user=polars password=s3cret host=h",  # libpq key-value form
    ],
)
def test_bad_connection_string_is_not_echoed(connection: str) -> None:
    with pytest.raises(ValueError, match="neither") as exc:
        pl.scan_database("t", connection)
    assert "s3cret" not in str(exc.value)


def test_odbc_string_in_any_key_order() -> None:
    lf = pl.scan_database(
        "t",
        "Server=h;Driver={ODBC Driver 18 for SQL Server};UID=u;PWD=s3cret",
        schema={"x": pl.Int64},
        dialect="mssql",
        driver="arrow_odbc",
    )
    try:
        plan = lf.explain()
    except pl.exceptions.PolarsError as e:  # the ODBC module may be missing
        plan = str(e)
    assert "s3cret" not in plan


def test_invalid_arguments(uri: str) -> None:
    def scan(**kwargs: Any) -> pl.LazyFrame:
        return pl.scan_database("t", uri, **kwargs)

    invalid: list[dict[str, Any]] = [
        {"partition_num": 0},
        {"partition_num": True},
        {"partition_num": -1},
        {"max_connections": 0},
        {"batch_size": 0},
        {"partition_on": None, "partition_num": 4},
        {"partition_on": None, "partition_range": (1, 10)},
        {"partition_range": (1, 10)},  # with "auto"
        {"partition_predicates": ["id < 5"], "partition_on": "id"},
        {"partition_predicates": ["id < 5"], "partition_num": 2},
        {"partition_on": "id", "partition_range": (100, 1)},
        {"consistency": "strict"},
        {"source_type": "view"},
        {"dialect": "not_a_dialect"},
    ]
    for kwargs in invalid:
        with pytest.raises(ValueError):
            scan(**kwargs)
    with pytest.raises(ValueError, match="source"):
        pl.scan_database("", uri)
    with pytest.raises(TypeError):
        scan(credential_provider=123)
    with pytest.raises(TypeError, match="cloud storage"):
        scan(credential_provider=pl.CredentialProviderAWS())


def test_partition_range_must_fit_the_column(uri: str) -> None:
    lf = pl.scan_database(
        "t",
        uri,
        partition_on="id",
        partition_range=(date(2020, 1, 1), date(2021, 1, 1)),
    )
    with pytest.raises(pl.exceptions.InvalidOperationError, match="numeric"):
        lf.collect()


def test_partition_on_column_named_auto(uri: str) -> None:
    lf = pl.scan_database("t", uri, partition_on=pl.col("auto"), partition_num=4)
    assert 'range on "auto"' in lf.explain()
    assert lf.collect().height == 100


def test_snapshot_reads_with_one_query(uri: str) -> None:
    lf = pl.scan_database("t", uri, consistency="snapshot", partition_num=4)
    assert "partitions: 1" in lf.explain()
    assert lf.collect().height == 100


def test_driver_pins(uri: str) -> None:
    assert pl.scan_database("t", uri, driver="SQLITE3").collect().height == 100
    with pytest.raises(pl.exceptions.InvalidOperationError, match="not a driver for"):
        pl.scan_database("t", uri, driver="psycopg").collect()
    with pytest.raises(pl.exceptions.InvalidOperationError, match="empty"):
        pl.scan_database("t", uri, driver=" ").collect()


def test_typed_options(uri: str) -> None:
    lf = pl.scan_database("t", uri, driver="sqlite3", connect_options={"timeout": 5})
    assert lf.collect().height == 100
    pytest.importorskip("sqlalchemy")
    lf = pl.scan_database(
        "t", uri, driver="sqlalchemy", connect_options={"echo": False}
    )
    assert lf.collect().height == 100


def test_connectorx_rejects_options(uri: str) -> None:
    pytest.importorskip("connectorx")
    lf = pl.scan_database("t", uri, driver="connectorx", execute_options={"x": 1})
    with pytest.raises(pl.exceptions.ComputeError, match="connectorx takes no options"):
        lf.collect()


def test_connection_objects_take_no_connect_settings(db: Path) -> None:
    def factory() -> sqlite3.Connection:
        return sqlite3.connect(db, check_same_thread=False)

    with pytest.raises(ValueError, match="driver"):
        pl.scan_database("t", factory, driver="adbc")
    with pytest.raises(ValueError, match="connect_options"):
        pl.scan_database("t", factory, connect_options={"timeout": 1})
    with (
        closing(factory()) as conn,
        pytest.raises(ValueError, match="credential_provider"),
    ):
        pl.scan_database("t", conn, credential_provider=lambda: {"user": "u"})


def test_live_sqlite3_connection(db: Path) -> None:
    with (
        closing(sqlite3.connect(db)) as conn,
        pytest.raises(ValueError, match="check_same_thread"),
    ):
        pl.scan_database("t", conn)
    with closing(sqlite3.connect(db, check_same_thread=False)) as conn:
        assert pl.scan_database("t", conn).collect().height == 100


def test_sqlalchemy_engine_connects_through_its_pool() -> None:
    sqlalchemy = pytest.importorskip("sqlalchemy")
    from sqlalchemy.pool import StaticPool

    # an in-memory database that only this engine's single connection holds
    engine = sqlalchemy.create_engine(
        "sqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE m (id INTEGER)")
        conn.exec_driver_sql("INSERT INTO m VALUES (1), (2)")
    try:
        assert pl.scan_database("m", engine).collect()["id"].to_list() == [1, 2]
    finally:
        engine.dispose()


def test_credential_provider_without_user_or_password(uri: str) -> None:
    lf = pl.scan_database(
        "t", uri, driver="sqlite3", schema={"id": pl.Int64}, credential_provider=dict
    )
    with pytest.raises(pl.exceptions.ComputeError, match="neither 'user' nor"):
        lf.collect()


def test_params_keep_percent_literals_in_polars(uri: str) -> None:
    lf = pl.scan_database(
        "SELECT id, s FROM t WHERE id < ?", uri, params=[50], driver="sqlite3"
    ).filter(pl.col("s").str.starts_with("v1"), pl.col("s") != "50%")
    plan = lf.explain()
    assert "LIKE" not in plan
    assert "'50%'" not in plan
    assert sorted(lf.collect()["id"]) == [1, *range(10, 20)]


def test_query_that_picks_rows_is_not_partitioned(uri: str) -> None:
    for query in [
        "SELECT * FROM t LIMIT 10",
        "SELECT * FROM t ORDER BY id LIMIT 10 OFFSET 5",
        "SELECT * FROM t WHERE id > (SELECT id FROM t LIMIT 1)",
        "SELECT id, random() AS r FROM t",
    ]:
        lf = pl.scan_database(query, uri, partition_on="id", partition_num=2)
        with pytest.raises(pl.exceptions.InvalidOperationError, match="partitioned"):
            lf.collect()
        assert pl.scan_database(query, uri).collect().height > 0


def test_arrow_type_polars_cannot_read_is_an_error(tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute("CREATE TABLE t AS SELECT INTERVAL 1 DAY AS iv")
    lf = pl.scan_database("SELECT iv FROM t", f"duckdb:///{path}")
    with pytest.raises(pl.exceptions.ComputeError, match="can't read"):
        lf.collect()
