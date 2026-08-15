"""FastAPI surface for the local GlassEye dashboard."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .demo import DemoRunner, load_latest_demo
from .events import EventLog
from .paths import artifacts_root, repo_root
from .replay import replay_log

app = FastAPI(
    title="GlassEye API",
    version="0.1.0",
    description="Local deterministic facade inspection and remediation simulator.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
artifacts_root().mkdir(parents=True, exist_ok=True)
app.mount("/artifacts", StaticFiles(directory=artifacts_root()), name="artifacts")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "glasseye"}


@app.get("/api/demo/latest")
def demo_latest() -> dict:
    result = load_latest_demo()
    if result is None:
        raise HTTPException(status_code=404, detail="No demo has run yet.")
    return result.model_dump(mode="json")


@app.post("/api/demo/run")
def demo_run() -> dict:
    """Run actual YOLO inference; never synthesize a result at the API boundary."""

    return DemoRunner().run().model_dump(mode="json")


@app.get("/api/missions/{mission_id}/events")
def mission_events(mission_id: str) -> dict:
    result = load_latest_demo()
    if result is None or result.mission_id != mission_id:
        raise HTTPException(status_code=404, detail="Unknown mission.")
    events = EventLog.load(repo_root() / result.event_log_ref)
    return {"mission_id": mission_id, "events": [event.model_dump(mode="json") for event in events]}


@app.get("/api/missions/{mission_id}/replay")
def mission_replay(mission_id: str) -> dict:
    result = load_latest_demo()
    if result is None or result.mission_id != mission_id:
        raise HTTPException(status_code=404, detail="Unknown mission.")
    return replay_log(repo_root() / result.event_log_ref).model_dump(mode="json")
