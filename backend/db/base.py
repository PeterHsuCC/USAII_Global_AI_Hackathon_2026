"""SQLAlchemy engine/session setup.

Storage technology decision (v6 Section 3 / 15.11): PostgreSQL is the
documented system of record. This deployment substitutes SQLite -- the
schema (models.py) is written with portable SQLAlchemy types (Uuid,
DateTime(timezone=True), Numeric) so it can be pointed at Postgres later
with a connection-string change, not a schema rewrite.
"""

from collections.abc import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.config import settings


class Base(DeclarativeBase):
    pass


def register_sqlite_foreign_keys(engine) -> None:
    """SQLite does not enforce FK constraints by default; the doc's "native
    foreign keys" requirement (Section 15.11) needs this set per-connection.
    Exposed so tests can apply it to their own isolated engines too.
    """

    @event.listens_for(engine, "connect")
    def _enable_sqlite_foreign_keys(dbapi_connection, connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


_is_sqlite = settings.database_url.startswith("sqlite")
_connect_args = {"check_same_thread": False} if _is_sqlite else {}

engine = create_engine(settings.database_url, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

if _is_sqlite:
    register_sqlite_foreign_keys(engine)


def create_all() -> None:
    Base.metadata.create_all(bind=engine)


def get_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()