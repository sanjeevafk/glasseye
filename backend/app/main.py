from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .demo import DemoRunner, load_latest_demo
from .detector import YoloDetector
from .events import EventLog
from .image_inspector import _get_model_checkpoint, inspect_image_bytes
from .paths import artifacts_root, frontend_dist, repo_root, samples_root
from .replay import replay_log


class CachedStaticFiles(StaticFiles):
    def __init__(self, *args, cache_control: str = "public, max-age=86400", **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_control = cache_control

    def file_response(self, *args, **kwargs) -> Response:
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = self.cache_control
        return resp


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm YOLO model and constrain PyTorch CPU threads for 512 MB Render Free Tier
    try:
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        checkpoint_path, _ = _get_model_checkpoint()
        detector = YoloDetector(checkpoint_path, confidence=0.15, image_size=320)
        detector.load()
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        detector.predict_frame(dummy, frame_id="warmup", timestamp=0.0, image_id="warmup")
    except Exception:
        pass
    yield


app = FastAPI(
    title="GlassEye API",
    version="0.1.0",
    description="Deterministic facade inspection and remediation simulator.",
    lifespan=lifespan,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
# Same-origin production (frontend built into frontend/dist and served by this
# app) needs no CORS; the dev split (Vite on :5173) is allowed for local work.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "HEAD"],
    allow_headers=["Content-Type"],
)
artifacts_root().mkdir(parents=True, exist_ok=True)
samples_root().mkdir(parents=True, exist_ok=True)
app.mount(
    "/artifacts",
    CachedStaticFiles(directory=artifacts_root(), cache_control="public, max-age=3600"),
    name="artifacts",
)
app.mount(
    "/samples",
    CachedStaticFiles(directory=samples_root(), cache_control="public, max-age=86400"),
    name="samples",
)


@app.api_route("/health", methods=["GET", "HEAD"])
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


@app.post("/api/inspect/image")
async def inspect_custom_image(
    file: Annotated[UploadFile, File(...)],
    confidence: Annotated[float, Form()] = 0.15,
    model_choice: Annotated[str | None, Form()] = None,
    run_vlm: Annotated[bool, Form()] = True,
) -> dict:
    """Run real-time YOLO defect detection, panel localization, severity scoring,
    policy suggestions, and optional advisory VLM review on a custom user image."""
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if len(content) > 15 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image file exceeds 15 MB limit.")

        result = inspect_image_bytes(
            content,
            filename=file.filename or "uploaded_facade.jpg",
            confidence=confidence,
            model_choice=model_choice,
            run_vlm=run_vlm,
        )
        return result.model_dump(mode="json")
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image inspection failed: {exc}") from exc


@app.get("/api/inspect/samples")
def get_sample_images() -> list[dict]:
    """Return available preset sample facade images for quick testing."""
    samples_dir = samples_root()
    if not any(samples_dir.glob("*.jpg")) and (artifacts_root() / "samples").is_dir():
        samples_dir = artifacts_root() / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    results = []
    metadata = {
        "structural_crack_sample.jpg": {
            "title": "Concrete Facade Fracture",
            "description": "High-severity concrete crack along panel seam",
            "expected_type": "structural",
        },
        "spalling_damage_sample.jpg": {
            "title": "Severe Spalling Defect",
            "description": "Exposed aggregate and concrete delamination",
            "expected_type": "structural",
        },
        "clean_facade_sample.jpg": {
            "title": "Nominal Building Facade",
            "description": "Clean composite panels with intact grout lines",
            "expected_type": "nominal",
        },
    }
    for file in sorted(samples_dir.glob("*.jpg")):
        info = metadata.get(
            file.name,
            {
                "title": file.stem.replace("_", " ").title(),
                "description": "Facade inspection test sample",
                "expected_type": "sample",
            },
        )
        url_path = f"/samples/{file.name}" if samples_dir == samples_root() else f"/artifacts/samples/{file.name}"
        results.append(
            {
                "filename": file.name,
                "url": url_path,
                **info,
            }
        )
    return results


if frontend_dist().is_dir():
    # Serve the built SPA in production (single port).  Registered last so API
    # and artifact routes keep priority; unknown paths fall back to index.html
    # for client-side routing.
    app.mount(
        "/assets",
        CachedStaticFiles(
            directory=frontend_dist() / "assets",
            cache_control="public, max-age=31536000, immutable",
        ),
        name="assets",
    )

    @app.api_route("/", methods=["GET", "HEAD"], response_model=None)
    def index() -> FileResponse:
        return FileResponse(
            frontend_dist() / "index.html",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    @app.get("/{path:path}", response_model=None)
    def spa_fallback(path: str) -> FileResponse:
        candidate = frontend_dist() / path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(
            frontend_dist() / "index.html",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
