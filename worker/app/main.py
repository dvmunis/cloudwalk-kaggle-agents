"""HTTP API the n8n agents call. Every endpoint is deterministic; decisions live in the agents.

Requests that carry an `X-Attempt-Id` header are recorded in that attempt's
trace (tool call + result or error), so agents only need to log their own
reasoning explicitly.
"""

import logging
import traceback
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse

from . import external, kaggle_client, leaderboard, metrics, profiling, settings, store, trace, training
from .schemas import (AttemptCreate, AttemptFinish, DiscoverRequest, EventCreate, ExecutionFailure, SubmitRequest,
                      TrainConfig)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("worker")



@asynccontextmanager
async def lifespan(_: FastAPI):
    for attempt_id in trace.sweep_stale():
        log.warning("attempt %s closed as orphaned", attempt_id)
    yield


app = FastAPI(title="Kaggle ML Worker", version="0.2.0", lifespan=lifespan)

_loaded: dict[str, profiling.CompetitionData] = {}


def _data(slug: str) -> profiling.CompetitionData:
    if slug not in _loaded:
        _loaded[slug] = profiling.load(slug)
    return _loaded[slug]


def _traced(attempt_id: str | None, tool: str, args: dict, fn: Callable[[], Any]) -> Any:
    if not attempt_id:
        return fn()
    trace.log_event(attempt_id, "worker", "tool_call", {"tool": tool, "args": args})
    try:
        result = fn()
    except Exception as exc:
        error = {"tool": tool, "error": str(exc)}
        # Agent mistakes and Kaggle refusals are explained by their message; anything
        # else is a bug, and the trace keeps where it happened.
        if not isinstance(exc, (ValueError, kaggle_client.KaggleError)):
            error["traceback"] = traceback.format_exc(limit=8)
        trace.log_event(attempt_id, "worker", "error", error)
        raise
    trace.log_event(attempt_id, "worker", "tool_result", {"tool": tool, "result": result})
    return result


# Errors are returned as plain messages so an agent can read them and adjust its next call.
@app.exception_handler(ValueError)
async def value_error(request: Request, exc: ValueError):
    log.info("422 %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(status_code=422, content={"error": str(exc)})


@app.exception_handler(kaggle_client.KaggleError)
async def kaggle_error(request: Request, exc: kaggle_client.KaggleError):
    log.warning("502 %s %s: Kaggle: %s", request.method, request.url.path, exc)
    return JSONResponse(status_code=502, content={"error": f"Kaggle: {exc}"})


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception):
    log.exception("500 %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse(status_code=500, content={"error": f"{type(exc).__name__}: {exc}"})


@app.get("/health")
def health():
    return {"status": "ok", "supported_metrics": metrics.SUPPORTED}


# --- Competition tools -------------------------------------------------------

@app.post("/competitions/{slug}/download")
def download(slug: str, x_attempt_id: str | None = Header(None)):
    def run():
        files = kaggle_client.download(slug, settings.competition_data_dir(slug))
        _loaded.pop(slug, None)
        return {"competition": slug, "files": files}

    return _traced(x_attempt_id, "download", {"competition": slug}, run)


@app.get("/competitions/{slug}/profile")
def profile(slug: str, x_attempt_id: str | None = Header(None)):
    def run():
        return {**profiling.profile(_data(slug)), "external_datasets": external.available(slug)}

    return _traced(x_attempt_id, "profile", {"competition": slug}, run)


@app.post("/competitions/{slug}/external/discover")
def discover_external(slug: str, req: DiscoverRequest, x_attempt_id: str | None = Header(None)):
    return _traced(x_attempt_id, "discover_external", {"competition": slug, **req.model_dump()},
                   lambda: external.discover(_data(slug), req.queries))


@app.get("/competitions/{slug}/external")
def external_datasets(slug: str):
    return external.available(slug)


@app.post("/competitions/{slug}/train")
def train(slug: str, cfg: TrainConfig, x_attempt_id: str | None = Header(None)):
    return _traced(x_attempt_id, "train", cfg.model_dump(), lambda: training.train(_data(slug), cfg, x_attempt_id))


@app.post("/competitions/{slug}/submit")
def submit(slug: str, req: SubmitRequest, x_attempt_id: str | None = Header(None)):
    def run():
        file = store.run_dir(slug, req.run_id) / "submission.csv"
        description = f"{req.run_id} {req.message}".strip()
        kaggle_client.submit(slug, file, description)
        if not req.wait_for_score:
            return {"run_id": req.run_id, "status": "submitted"}
        scored = kaggle_client.wait_for_score(slug, description)
        return {"run_id": req.run_id, "status": "complete", "public_score": float(scored["publicScore"])}

    return _traced(x_attempt_id, "submit", req.model_dump(), run)


@app.get("/competitions/{slug}/leaderboard")
def leaderboard_position(slug: str, x_attempt_id: str | None = Header(None)):
    def run():
        rows = kaggle_client.leaderboard(slug, settings.competition_runs_dir(slug) / "_leaderboard")
        return {"competition": slug, **leaderboard.summarize(rows, kaggle_client.submissions(slug))}

    return _traced(x_attempt_id, "leaderboard", {"competition": slug}, run)


@app.get("/competitions/{slug}/experiments")
def experiments(slug: str):
    return store.experiments(slug)


@app.get("/competitions/{slug}/history")
def history(slug: str, limit: int = 10):
    return store.history(slug, limit)


# --- Attempt trace -----------------------------------------------------------

@app.post("/attempts")
def start_attempt(req: AttemptCreate):
    return trace.start_attempt(req.competition, req.goal, req.execution_id)


@app.post("/executions/{execution_id}/fail")
def fail_execution(execution_id: str, req: ExecutionFailure):
    return trace.fail_execution(execution_id, req.node, req.error)


@app.post("/attempts/{attempt_id}/events")
def log_event(attempt_id: str, event: EventCreate):
    trace.log_event(attempt_id, event.agent, event.kind, event.content, event.reasoning, event.context, event.meta)
    return {"ok": True}


@app.post("/attempts/{attempt_id}/finish")
def finish_attempt(attempt_id: str, req: AttemptFinish):
    return trace.finish_attempt(attempt_id, req.status, req.summary)


@app.get("/attempts")
def list_attempts(competition: str | None = None):
    trace.sweep_stale()
    return trace.list_attempts(competition)


@app.get("/attempts/{attempt_id}")
def get_attempt(attempt_id: str):
    trace.sweep_stale()
    return trace.get_attempt(attempt_id)
