"""Shared FastAPI dependencies: DB session and request correlation id."""

import uuid
from collections.abc import Iterator

from sqlalchemy.orm import Session

from backend.db.base import SessionLocal


def get_db() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def new_request_id() -> uuid.UUID:
    return uuid.uuid4()