"""FastAPI application entrypoint.

Run from the repo root:
    uvicorn backend.main:app --reload
"""

import logging
from contextlib import asynccontextmanager

import backend.paths  # noqa: F401 -- must run before any risk_detection import
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routers import admin, audit, auth, cases, review
from backend.config import settings
from backend.db.base import create_all
from backend.jobs.queue import job_queue
from backend.model_runtime.loader import get_model_components
from backend.monitoring.logging_setup import bind_request_id, configure_logging

log = logging.getLogger("risk_platform")


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    create_all()

    log.info("Loading model components (mode=%s)", settings.model_runtime_mode)
    get_model_components()

    job_queue.start()
    replayed = job_queue.replay_unfinished_jobs()
    log.info("Replayed %d unfinished job(s) on startup", replayed)

    yield

    await job_queue.stop()


app = FastAPI(title="Conversation Risk Decision Support System", lifespan=lifespan)

# Hackathon MVP: permissive CORS. Section 11 names CORS/security headers as
# API-layer responsibilities that need real configuration before production
# (specific allowed origins, not "*").
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = bind_request_id()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """FastAPI's default 422 handler echoes each rejected field's raw
    `"input"` value back in the response body -- for POST /cases, that can
    be the unredacted conversation text the analyst just submitted (a
    sibling field failing validation, e.g. a bad timestamp, still surfaces
    the message text next to it in the same error list). Strip `input`
    from every error entry before it leaves the server."""
    sanitized = [{k: v for k, v in error.items() if k != "input"} for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": sanitized})


app.include_router(auth.router)
app.include_router(cases.router)
app.include_router(review.router)
app.include_router(audit.router)
app.include_router(admin.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}