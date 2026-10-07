"""FastAPI application entrypoint (Project.md #10, #63)."""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import OperationalError

from app.api import assets, auth, evidence, inspection, markers, metrology, models3d, procedures, video, vision
from app.core.config import get_settings

app = FastAPI(
    title="TVASTA Industrial AR/MR Procedure & QA Platform",
    version="0.1.0",
    description="Phase 1-5 — camera-based procedure execution, deterministic QA, "
    "3D viewer, marker-based physical<->3D registration, tracking, and video-to-procedure extraction.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(OperationalError)
def database_unreachable(request: Request, exc: OperationalError) -> JSONResponse:
    """Database down (e.g. Docker Desktop not started): a clear 503 instead of a
    bare 500. A 500 is sent without CORS headers, so the browser reported it as
    "blocked by CORS policy", which hid the real cause."""
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Database not reachable — start Docker Desktop and the tvasta-postgres container, "
            f"then retry. ({type(exc.orig).__name__ if exc.orig else 'OperationalError'})"
        },
    )


app.include_router(auth.router)
app.include_router(assets.router)
app.include_router(procedures.router)
app.include_router(inspection.router)
app.include_router(vision.router)
app.include_router(evidence.router)
app.include_router(models3d.router)
app.include_router(video.router)
app.include_router(metrology.router)
app.include_router(markers.router)

# Serves GLB/glTF files referenced by Model3D.storage_key (Project.md #20,
# Phase 4). A later phase can move this behind MinIO/S3 without changing the
# API contract — the frontend only ever sees a URL.
_settings = get_settings()
_models_dir = Path(_settings.models_3d_dir)
_models_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/models", StaticFiles(directory=str(_models_dir)), name="models_3d")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
