import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db import models  # noqa: F401 -- registers tables on Base.metadata
from backend.db.base import Base, register_sqlite_foreign_keys


@pytest.fixture()
def db_sessionmaker():
    """Fresh in-memory SQLite DB per test -- isolated from the dev .db file.

    Yields a sessionmaker (not a single Session) so code under test that
    opens its own sessions per call (e.g. AnalysisJobQueue) can be pointed
    at this isolated DB too.

    Uses SQLite's URI shared-cache mode rather than `:memory:` +
    StaticPool: tests routinely have more than one thread touching the DB
    at once (FastAPI runs every router's sync path operation function in
    a worker thread via Starlette's threadpool, and the job queue's
    worker runs in another thread via asyncio.to_thread), and StaticPool
    means literally one shared DBAPI connection object handed out to
    every thread -- not safe to operate on concurrently, and previously
    caused a committed status change to appear to vanish to the other
    thread. An attempt to fix that by serializing checkout/checkin with a
    lock deadlocked instead (checkout/checkin aren't reliably 1:1 paired
    per logical session). Shared-cache mode sidesteps the problem
    entirely: each thread gets its own real connection from a normal
    pool, and SQLite's own (connection-level, not Python-level) locking
    correctly serializes access to the underlying shared data, the same
    way a real multi-connection Postgres/file-SQLite deployment would.
    """
    db_name = f"file:memdb_{uuid.uuid4().hex}?mode=memory&cache=shared&uri=true"
    engine = create_engine(f"sqlite:///{db_name}", connect_args={"check_same_thread": False})
    register_sqlite_foreign_keys(engine)

    # Shared-cache in-memory DBs are destroyed once their last connection
    # closes; holding one open for the fixture's lifetime keeps it alive
    # while the sessionmaker's pooled connections cycle in and out.
    keepalive_connection = engine.connect()

    Base.metadata.create_all(bind=engine)

    TestSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    try:
        yield TestSessionLocal
    finally:
        keepalive_connection.close()
        engine.dispose()


@pytest.fixture()
def db_session(db_sessionmaker):
    session = db_sessionmaker()
    try:
        yield session
    finally:
        session.close()