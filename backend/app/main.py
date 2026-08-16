"""FastAPI surface for the local GlassEye dashboard."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .demo import DemoRunner, load_latest_demo
from .events import EventLog
from .paths import artifacts_root, frontend_dist, repo_root
from .replay import replay_log

app = FastAPI(
    title="GlassEye API",
    version="0.1.0",
    description="Deterministic facade inspection and remediation simulator.",
)

# Same-origin production (frontend built into frontend/dist and served by this
# app) needs no CORS; the dev split (Vite on :5173) is allowed for local work.
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
    """Return the deterministic demo result.

    The demo is fully seeded (fixed scenario seed, fixed model, fixed media),
    so every run produces an identical result. In the deployed image the demo
    is baked at build time; serving that precomputed result is honest (it IS
    the real YOLO inference output) and makes the button instant on slow
    free-tier CPU. When no baked result exists (local dev before `make demo`),
    run the pipeline for real.
    """

    baked = load_latest_demo()
    if baked is not None:
        return baked.model_dump(mode="json")
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


if frontend_dist().is_dir():
    # Serve the built SPA in production (single port).  Registered last so API
    # and artifact routes keep priority; unknown paths fall back to index.html
    # for client-side routing.
    app.mount(
        "/assets",
        StaticFiles(directory=frontend_dist() / "assets"),
        name="assets",
    )

    @app.get("/", response_model=None)
    def index() -> FileResponse:
        return FileResponse(frontend_dist() / "index.html")

    @app.get("/{path:path}", response_model=None)
    def spa_fallback(path: str) -> FileResponse:
        candidate = frontend_dist() / path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(frontend_dist() / "index.html")
